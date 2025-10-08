import torch
from pathlib import Path
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. Configuration ---

# Model and Adapter Paths 
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER_DIR = Path("./phi3_oski_adapter")

# RAG Configuration 
CHROMA_PERSIST_DIR = Path("./chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. Setup LLM (Base Model + LoRA Adapter) ---

# --- 2. Setup LLM (Base Model + LoRA Adapter) ---

def load_fine_tuned_model():
    """Loads the base Phi-3 model and applies the LoRA adapter weights."""
    print("Loading base model and 4-bit quantization config...")
    
    # Setup 4-bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # Load tokenizer
    # *** CRITICAL FIX: Removed trust_remote_code=True ***
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR)
    tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Set padding side to left for stable generation
    tokenizer.padding_side = "left" 

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        # *** CRITICAL FIX: Removed trust_remote_code=True ***
        # trust_remote_code=True, 
        # CRITICAL FIX for 'DynamicCache' error: Disable cache at load time
        use_cache=False, 
    )

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    model.eval()
    
    # Ensure the final model config also reflects use_cache=False
    # (This is the "Bulletproof Code Fix" recommended for this error)
    model.config.use_cache = False 
    
    print(f"✅ Phi-3 OskiBot loaded successfully from {LORA_ADAPTER_DIR}")
    return model, tokenizer

# --- The rest of the script remains the same ---

# --- 3. Setup RAG Retriever ---

def load_rag_retriever():
    """Loads the ChromaDB vector store and creates a retriever instance."""
    print("Loading RAG vector database (ChromaDB)...")
    
    if not CHROMA_PERSIST_DIR.exists():
        print(f"Error: ChromaDB directory not found at {CHROMA_PERSIST_DIR}")
        print("Please run '1_create_rag_index.py' first.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR), 
        embedding_function=embeddings
    )
    
    # Retrieve top 3 documents
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("✅ RAG Retriever ready.")
    return retriever

# --- 4. RAG Orchestration and Prompt Generation ---

def generate_augmented_prompt(retriever, conversation_history, tokenizer):
    """
    1. Retrieves context from RAG.
    2. Builds the final, structured prompt with context and history.
    """
    latest_query = conversation_history[-1]['content']
    
    # --- RAG Step: Retrieval ---
    retrieved_docs = retriever.invoke(latest_query)
    
    context_chunks = [doc.page_content for doc in retrieved_docs]
    context_string = "\n---\n".join(context_chunks)

    # --- Augmentation Step: Construct the final Phi-3 prompt ---
    
    # System instruction based on the persona defined in your fine-tuning data
    system_instruction = (
        "You are OskiBot, a helpful, enthusiastic tutor for UC Berkeley Golden Bears "
        "history and football. Your primary goal is to guide the student conceptually "
        "or refer them to the provided context before giving a direct answer. "
        "Always encourage them with 'Go Bears!'."
    )
    
    # Inject the RAG context into the system prompt
    context_template = f"\n\n--- RAG Context ---\n{context_string}\n--- End RAG Context ---"
    
    full_system_prompt = f"{system_instruction}{context_template}"
    
    # Full conversation list for the chat template
    full_conversation = [
        {"role": "system", "content": full_system_prompt}
    ] + conversation_history
    
    # Apply the Phi-3 ChatML template
    final_prompt = tokenizer.apply_chat_template(
        full_conversation, 
        tokenize=False, 
        add_generation_prompt=True 
    )
    
    return final_prompt

# --- 5. Chat Loop and Inference ---

def run_chat_loop(model, tokenizer, retriever):
    """Runs the main interactive chat session."""
    conversation_history = []
    
    print("\n--- OskiBot MWE Chat Started ---")
    print("Type 'quit' or 'exit' to end the session. Go Bears!")
    print("-" * 35)

    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("OskiBot: Go Bears! Study hard!")
            break

        if not user_input.strip():
            continue

        # 1. Update history with the new user message
        conversation_history.append({"role": "user", "content": user_input})
        
        # 2. Generate the RAG-augmented, conversational prompt
        final_prompt = generate_augmented_prompt(retriever, conversation_history, tokenizer)

        # 3. Model Inference (Generation)
        try:
            # Tokenize with padding and truncation
            inputs = tokenizer(
                final_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=tokenizer.model_max_length
            ).to(model.device)
            
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Generate the response
            # use_cache=False is now enforced by model config, avoiding the 'seen_tokens' error.
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode only the newly generated part 
            start_index = input_ids.shape[-1]
            response_text = tokenizer.decode(output[0, start_index:], skip_special_tokens=True)
            
            # 4. Update history with the assistant response
            conversation_history.append({"role": "assistant", "content": response_text})

            print(f"OskiBot: {response_text.strip()}")

        except Exception as e:
            print(f"An error occurred during generation: {e}")
            # Remove the last user input to allow retry
            if conversation_history:
                conversation_history.pop()


# --- Main Execution ---

if __name__ == "__main__":
    if not LORA_ADAPTER_DIR.exists() or not CHROMA_PERSIST_DIR.exists():
        print(f"Missing fine-tuning or RAG index directories.")
        print("Please run '1_create_rag_index.py' and '2_fine_tune_phi3.py' first.")
        sys.exit(1)
    
    try:
        # Load all components
        phi3_model, phi3_tokenizer = load_fine_tuned_model()
        rag_retriever = load_rag_retriever()
        
        # Start the interactive chat
        if rag_retriever:
            run_chat_loop(phi3_model, phi3_tokenizer, rag_retriever)
    except Exception as e:
        print(f"Fatal Error during initialization: {e}")
        sys.exit(1)
