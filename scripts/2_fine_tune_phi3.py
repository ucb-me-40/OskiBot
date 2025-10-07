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
TRAINING_DATA_FILE = "data/fine_tune_conversations.jsonl" 

# QLoRA/Training Parameters
compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LORA_R = 64       
LORA_ALPHA = 16   
LORA_DROPOUT = 0.1 
MAX_SEQ_LENGTH = 1024 

# Training Arguments 
TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,               
    per_device_train_batch_size=2,    
    gradient_accumulation_steps=4,    
    logging_steps=20,
    learning_rate=2e-4,               
    optim="paged_adamw_8bit",         
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    save_strategy="epoch",
)

# --- 2. Setup Quantization and Model/Tokenizer Loading ---

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

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
tokenizer.padding_side = "right" 

# --- 3. Setup LoRA Configuration ---

peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM", 
    target_modules="all-linear", 
)

# --- 4. Load and Prepare Conversational Data ---

try:
    dataset = load_dataset("json", data_files=TRAINING_DATA_FILE, split="train")
    print(f"Loaded {len(dataset)} training examples.")
except Exception as e:
    print(f"Error loading data: {e}. Ensure '{TRAINING_DATA_FILE}' is correctly formatted.")
    exit()

# --- 5. Initialize and Start the Trainer ---

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=TRAINING_ARGS,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text", 
    packing=False,
)

logging.set_verbosity_warning()

print("\n--- Starting Fine-Tuning ---")
trainer.train()
print("--- Fine-Tuning Complete! ---")

# --- 6. Save the Adapter Weights ---

trainer.model.save_pretrained(LORA_ADAPTER_DIR)
tokenizer.save_pretrained(LORA_ADAPTER_DIR)
print(f"✅ LoRA adapter weights saved to {LORA_ADAPTER_DIR}")