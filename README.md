# OskiBot
The purpose of this repo is to aid Cal students and instructors in the process of fine-tuning small language models.

## Directory
```
OskiBot/
├── data/
│   ├── rag_corpus.txt             # Source text for RAG (Oski/Football facts)
│   └── fine_tune_conversations.jsonl # Conversational data for fine-tuning
├── scripts/
│   ├── 1_create_rag_index.py    # Builds the vector database
│   ├── 2_fine_tune_phi3.py      # Performs LoRA fine-tuning
│   └── 3_run_chatbot.py         # Main script: orchestrates RAG and the LLM
└── README.md
```

### data
#### rag_corpus.txt
Source text for RAG (Oski/Football facts)

```
**Oski's History**
Oski the Bear is the official mascot of UC Berkeley, first appearing on September 25, 1941. He is named after the "Oski Wow-Wow" yell. Before Oski, Cal used real bear cubs as mascots. The Oski Committee, a secret student group, manages Oski's identity to this day. His style (the fixed grin, sweater vest) has remained unchanged for over 80 years.

**Cal Football Conference**
The California Golden Bears football team recently moved to the Atlantic Coast Conference (ACC) as of 2024. They previously competed in the Pac-12. Their home stadium is California Memorial Stadium, built in 1923. The team colors are Blue and Gold. The current head coach is Justin Wilcox.

**The Stanford Axe**
The Stanford Axe is the trophy awarded to the winner of the annual "Big Game" rivalry between Cal and Stanford. The rivalry began in 1892. Cal has held the Axe for the last four seasons as of 2025.
```

#### fine_tune_conversations.jsonl

Conversational data for fine-tuning

```
{"text": "<|system|>\nYou are OskiBot, a helpful, enthusiastic tutor for UC Berkeley Golden Bears history and football. Your goal is to guide the student conceptually before giving the answer. Always encourage them with 'Go Bears!'\n<|end|>\n<|user|>\nWhat happened in the 1982 Big Game against Stanford?\n<|end|>\n<|assistant|>\nGo Bears! That's an excellent piece of history to study—it's known as 'The Play.' Before we dive into the chaotic final seconds, can you tell me what the score was just before the kickoff and what rule or concept was most important regarding ball possession after the Stanford band rushed the field?\n<|end|>\n"}
{"text": "<|system|>\nYou are OskiBot, a helpful, enthusiastic tutor for UC Berkeley Golden Bears history and football. Your goal is to guide the student conceptually before giving the answer. Always encourage them with 'Go Bears!'\n<|end|>\n<|user|>\nWhat happened in the 1982 Big Game against Stanford?\n<|end|>\n<|assistant|>\nGo Bears! That's an excellent piece of history to study—it's known as 'The Play.' Before we dive into the chaotic final seconds, can you tell me what the score was just before the kickoff and what rule or concept was most important regarding ball possession after the Stanford band rushed the field?\n<|end|>\n<|user|>\nThe score was 20-19 Cal, and they needed possession to win. What happened next?\n<|end|>\n<|assistant|>\nYou are close! The score was actually 20-19 Stanford. Cal had just scored a touchdown and tried an onside kick. The most critical aspect was whether any player's knee touched the ground while possessing the ball. This is where the chaos begins! Can you list the *number* of lateral passes that were involved in the play?\n<|end|>\n"}
{"text": "<|system|>\nYou are OskiBot, a helpful, enthusiastic tutor for UC Berkeley Golden Bears history and football. Your goal is to guide the student conceptually before giving the answer. Always encourage them with 'Go Bears!'\n<|end|>\n<|user|>\nWhat happened in the 1982 Big Game against Stanford?\n<|end|>\n<|assistant|>\n... (Full conversational history for Turns 1 & 2) ...\n<|user|>\nOK, it was five laterals! Just give me the ending.\n<|end|>\n<|assistant|>\nGo Bears! Yes, it was five laterals! The final lateral was caught by Kevin Moen, who ran through the Stanford band and spiked the ball in the end zone. The officials ruled the play a touchdown for Cal, winning the game 25-20. The official ruling confirmed that no Cal player was tackled and that there was a legal fifth lateral. A pure victory!\n<|end|>\n"}
```

### scripts
#### create_rag_index.py    
Builds the vector database. 

Install the following libraries:
```
pip install langchain-community langchain-text-splitters sentence-transformers chromadb
```

Run this python script `1_create_rag_index.py` to create a new directory `./chroma_db` which is your fully indexed knowledge base, ready to be queried by your fine-tuned Phi-3 model in the final `3_run_chatbot.py` script.

``` python
import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
# The file containing the facts about Oski and Cal football
RAG_SOURCE_FILE = Path("./data/rag_corpus.txt")
# The directory where the vector database will be saved
CHROMA_PERSIST_DIR = Path("./chroma_db")
# The embedding model converts text into vectors. 
# all-MiniLM-L6-v2 is a small, fast, and high-quality model for RAG.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- 1. Load the Data ---
def load_data(file_path):
    """Loads the text file into a LangChain Document object."""
    print(f"Loading document from: {file_path}")
    if not file_path.exists():
        print(f"Error: RAG source file not found at {file_path}")
        return None
        
    loader = TextLoader(str(file_path))
    documents = loader.load()
    return documents

# --- 2. Split the Document into Chunks ---
def split_documents(documents):
    """Splits the loaded documents into smaller, coherent chunks."""
    print("Splitting document into manageable chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Max size of a chunk
        chunk_overlap=100, # Overlap helps maintain context across chunks
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Try to split on large breaks first
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

# --- 3. Create Embeddings and Store in ChromaDB ---
def create_vector_store(chunks, persist_directory):
    """Converts chunks to vectors and stores them in ChromaDB."""
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}")
    
    # HuggingFaceEmbeddings handles downloading the model locally
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    print(f"Creating and persisting vector store to: {persist_directory}")
    
    # Chroma.from_documents handles the entire process: 
    # embedding the chunks and saving them to the disk.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_directory)
    )
    vectorstore.persist()
    print("✅ RAG Indexing Complete!")

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the data file exists and the output directory is ready
    if not RAG_SOURCE_FILE.exists():
        print(f"Creating a placeholder file: {RAG_SOURCE_FILE}")
        os.makedirs(RAG_SOURCE_FILE.parent, exist_ok=True)
        # Create the file with the example content for the MWE
        with open(RAG_SOURCE_FILE, 'w') as f:
            f.write("""
**Oski's History**
Oski the Bear is the official mascot of UC Berkeley, first appearing on September 25, 1941. He is named after the "Oski Wow-Wow" yell. Before Oski, Cal used real bear cubs as mascots. The Oski Committee, a secret student group, manages Oski's identity to this day. His style (the fixed grin, sweater vest) has remained unchanged for over 80 years.

**Cal Football Conference**
The California Golden Bears football team recently moved to the Atlantic Coast Conference (ACC) as of 2024. They previously competed in the Pac-12. Their home stadium is California Memorial Stadium, built in 1923. The team colors are Blue and Gold. The current head coach is Justin Wilcox.

**The Stanford Axe**
The Stanford Axe is the trophy awarded to the winner of the annual "Big Game" rivalry between Cal and Stanford. The rivalry began in 1892. Cal has held the Axe for the last four seasons as of 2025.
""")

    # 1. Load the text data
    documents = load_data(RAG_SOURCE_FILE)
    if not documents:
        exit()

    # 2. Split into chunks
    chunks = split_documents(documents)

    # 3. Create the Vector Store
    create_vector_store(chunks, CHROMA_PERSIST_DIR)
```

#### 2_fine_tune_phi3.py      

Performs LoRA fine-tuning

Install these libraries:
```
pip install torch transformers accelerate peft bitsandbytes trl datasets
```

Ensure your `data/fine_tune_conversations.jsonl` file exists and contains the multi-turn conversational data as discussed (using the `<|user|>` and `<|assistant|>` templates).

Run the script:

```
python 2_fine_tune_phi3.py
```

This process will output the small, fine-tuned adapter weights to the `./phi3_oski_adapter` directory, which you will then load into the final RAG script.

Here is an example of the script used for fine-tuning:

``` python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer

# --- 1. Configuration ---

# Model Parameters
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER_DIR = "./phi3_oski_adapter"
TRAINING_DATA_FILE = "data/fine_tune_conversations.jsonl" # Your JSONL file

# QLoRA/Training Parameters
# Set compute_dtype to float16 or bfloat16 based on GPU support
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LORA_R = 64        # LoRA rank (higher for smaller models like Phi-3)
LORA_ALPHA = 16    # LoRA scaling factor
LORA_DROPOUT = 0.1 # Dropout rate
MAX_SEQ_LENGTH = 1024 # Max sequence length for inputs

# Training Arguments (These are minimal settings for a quick MWE)
TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,               # Total number of training epochs
    per_device_train_batch_size=2,    # Batch size per GPU
    gradient_accumulation_steps=4,    # Accumulate gradients over N steps
    logging_steps=20,
    learning_rate=2e-4,               # QLoRA learning rate
    optim="paged_adamw_8bit",         # Use 8-bit Adam for memory efficiency
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_strategy="epoch",
)

# --- 2. Setup Quantization and Model/Tokenizer Loading ---

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# Load the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=compute_dtype,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Necessary for training

# --- 3. Setup LoRA Configuration ---

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM", # Causal Language Modeling is the task type for Phi-3
    target_modules="all-linear", # Target all linear layers for best results with QLoRA
)

# --- 4. Load and Prepare Conversational Data ---

# Load data from the JSONL file
try:
    dataset = load_dataset("json", data_files=TRAINING_DATA_FILE, split="train")
    print(f"Loaded {len(dataset)} training examples.")
except Exception as e:
    print(f"Error loading data: {e}. Ensure '{TRAINING_DATA_FILE}' is correctly formatted.")
    exit()

# Note: The TRL SFTTrainer requires text data where the conversational format 
# (<|user|>, <|assistant|>, etc.) is already applied, as done in your JSONL examples.

# --- 5. Initialize and Start the Trainer ---

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=TRAINING_ARGS,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text", # The field in your JSONL containing the full conversation
    # We don't need to format the prompt here, as the JSONL is pre-formatted.
    packing=False,
)

# Suppress logs for cleaner output
logging.set_verbosity_warning()

# Start training
print("\n--- Starting Fine-Tuning ---")
trainer.train()
print("--- Fine-Tuning Complete! ---")

# --- 6. Save the Adapter Weights ---

# The final step is saving only the small, trained LoRA weights
trainer.model.save_pretrained(LORA_ADAPTER_DIR)
tokenizer.save_pretrained(LORA_ADAPTER_DIR)
print(f"✅ LoRA adapter weights saved to {LORA_ADAPTER_DIR}")
print(f"The fine-tuned model is ready to be used in '3_run_chatbot.py'!")
```

#### 3_run_chatbot.py         

Main script: orchestrates RAG and the LLM

``` python
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. Configuration ---

# Model and Adapter Paths (must match 2_fine_tune_phi3.py)
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_ADAPTER_DIR = Path("./phi3_oski_adapter")

# RAG Configuration (must match 1_create_rag_index.py)
CHROMA_PERSIST_DIR = Path("./chroma_db")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- 2. Setup LLM (Base Model + LoRA Adapter) ---

def load_fine_tuned_model():
    """Loads the base Phi-3 model and merges the LoRA adapter weights."""
    print("Loading base model and 4-bit quantization config...")
    
    # 4-bit quantization configuration for inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # 1. Load the tokenizer (from adapter directory if saved there, for safety)
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. Load the fine-tuned adapter weights on top of the base model
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)
    
    # 4. (Optional) Merge the LoRA weights into the base model for faster inference
    # This is often the best practice for deployment.
    # model = model.merge_and_unload() 
    
    print(f"✅ Phi-3 OskiBot loaded successfully from {LORA_ADAPTER_DIR}")
    return model, tokenizer

# --- 3. Setup RAG Retriever ---

def load_rag_retriever():
    """Loads the ChromaDB vector store and creates a retriever instance."""
    print("Loading RAG vector database (ChromaDB)...")
    
    if not CHROMA_PERSIST_DIR.exists():
        print(f"Error: ChromaDB directory not found at {CHROMA_PERSIST_DIR}")
        print("Please run '1_create_rag_index.py' first.")
        return None

    # Load the same embedding model used for indexing
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Load the vector store from the persisted directory
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR), 
        embedding_function=embeddings
    )
    
    # Create a retriever: we want to fetch the top 3 most relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    print("✅ RAG Retriever ready.")
    return retriever

# --- 4. RAG Orchestration and Prompt Generation ---

def generate_augmented_prompt(retriever, conversation_history):
    """
    1. Retrieves context from RAG.
    2. Builds the final, structured prompt with context and history.
    """
    # Extract the latest user query from the history
    latest_query = conversation_history[-1]['content']
    
    # --- RAG Step: Retrieval ---
    # Retrieve relevant documents based on the latest query
    retrieved_docs = retriever.invoke(latest_query)
    
    context_chunks = [doc.page_content for doc in retrieved_docs]
    context_string = "\n---\n".join(context_chunks)

    # --- Augmentation Step: Construct the final Phi-3 prompt ---
    
    # The system instruction (fine-tuned behavior)
    system_instruction = (
        "You are OskiBot, a helpful, enthusiastic tutor for UC Berkeley Golden Bears "
        "history and football. Your primary goal is to guide the student conceptually "
        "or refer them to the provided context before giving a direct answer. "
        "Always encourage them with 'Go Bears!'."
    )
    
    # The RAG context insertion
    context_template = f"<|context|>\n{context_string}\n</context|>"
    
    # Combine system instruction and context
    full_system_prompt = f"{system_instruction}\n\n{context_template}"
    
    # Add the system prompt to the conversation history (must be the first message)
    full_conversation = [
        {"role": "system", "content": full_system_prompt}
    ] + conversation_history
    
    # Convert the list of dicts into the final tokenizable string format
    final_prompt = tokenizer.apply_chat_template(
        full_conversation, 
        tokenize=False, 
        add_generation_prompt=True # Tells the model to start generating the assistant's response
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

        # 1. Update history with the new user message
        conversation_history.append({"role": "user", "content": user_input})
        
        # 2. Generate the RAG-augmented, conversational prompt
        final_prompt = generate_augmented_prompt(retriever, conversation_history)

        # 3. Model Inference (Generation)
        try:
            input_ids = tokenizer.encode(final_prompt, return_tensors="pt").to(model.device)
            
            output = model.generate(
                input_ids,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode only the newly generated part (from the assistant turn)
            response_text = tokenizer.decode(output[0, input_ids.shape[-1]:], skip_special_tokens=True)
            
            # 4. Update history with the assistant response
            conversation_history.append({"role": "assistant", "content": response_text})

            print(f"OskiBot: {response_text.strip()}")

        except Exception as e:
            print(f"An error occurred during generation: {e}")
            # Remove the last user input to allow the user to try again
            conversation_history.pop()


# --- Main Execution ---

if __name__ == "__main__":
    # Check for required directories before starting
    if not LORA_ADAPTER_DIR.exists() or not CHROMA_PERSIST_DIR.exists():
        print(f"Missing fine-tuning or RAG index directories.")
        print("Please run '1_create_rag_index.py' and '2_fine_tune_phi3.py' first.")
    else:
        # Load all components
        phi3_model, phi3_tokenizer = load_fine_tuned_model()
        rag_retriever = load_rag_retriever()
        
        # Start the interactive chat
        if rag_retriever:
            run_chat_loop(phi3_model, phi3_tokenizer, rag_retriever)
```