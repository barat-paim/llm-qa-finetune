import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback, BitsAndBytesConfig
from datasets import load_from_disk
from torch.utils.data import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def print_gpu_memory():
    print(f"GPU Memory: Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
          f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
# Step 1: Load the SQuAD dataset
squad_dataset = load_from_disk("./data/squad")

# Step 2: Model Setup
model_path = "./llama_3_2_1b_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Before model loading:")
print_gpu_memory()

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = prepare_model_for_kbit_training(model)

# Add LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Wrap the model with LoRA
model = get_peft_model(model, peft_config)

print("After model loading and LoRA configuration:")
print_gpu_memory()

# Ensure the tokenizer has a pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Step 3: Prepare the dataset
class SQuADDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        question = example['question']
        context = example['context']
        answer = example['answers']['text'][0] if example['answers']['text'] else ""

        # Construct input
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        target_text = f" {answer}"

        # Tokenize
        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        labels = targets.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Create datasets
train_dataset = SQuADDataset(squad_dataset['train'], tokenizer)
eval_dataset = SQuADDataset(squad_dataset['validation'], tokenizer)

# Step 4: Fine-Tuning Loop
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="steps",
    eval_steps=500,
    learning_rate=3e-4,  # Slightly higher learning rate for LoRA
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_steps=1000,
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
)

print("Before trainer initialization:")
print_gpu_memory()

class MemoryEfficientCallback(TrainerCallback):
    def __init__(self):
        self.total_loss = 0.0
        self.step_count = 0

    def on_step_end(self, args, state, control, **kwargs):
        if 'loss' in kwargs:
            self.total_loss += kwargs['loss'].item()
            self.step_count += 1

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.step_count > 0:
            avg_loss = self.total_loss / self.step_count
            print(f"Average loss for epoch: {avg_loss:.4f}")
        self.total_loss = 0.0
        self.step_count = 0

memory_efficient_callback = MemoryEfficientCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[MemoryEfficientCallback()]
)

print("After trainer initialization:")
print_gpu_memory()

# Start fine-tuning
print("Before training:")
print_gpu_memory()

trainer.train()

print("After training:")
print_gpu_memory()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_llama_squad")
tokenizer.save_pretrained("./fine_tuned_llama_squad")
