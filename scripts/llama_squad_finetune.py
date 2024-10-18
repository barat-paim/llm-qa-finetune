import torch
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.empty_cache()
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import psutil
import GPUtil
from tqdm.auto import tqdm
from torch.optim import Adam

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"GPU Memory: Allocated: {allocated:.2f} GB, "
          f"Reserved: {reserved:.2f} GB")
    
# Step 1: Load the SQuAD dataset
squad_dataset = load_from_disk("./data/squad")

# Step 2: Model Setup
model_path = "./llama_3_2_1b_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
print("Before model loading:")
print_gpu_memory()

# New: Use BitsAndBytesConfig for 8-bit quantization
from transformers import BitsAndBytesConfig

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

# New: Add LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# New: Wrap the model with LoRA
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

# Reduce dataset size
max_train_samples = 50000  # Adjust this number as needed
max_eval_samples = 5000   # Adjust this number as needed

if len(train_dataset) > max_train_samples:
    train_dataset = train_dataset.select(range(max_train_samples))
if len(eval_dataset) > max_eval_samples:
    eval_dataset = eval_dataset.select(range(max_eval_samples))

# New: Optimized data loading function
def create_dataloaders(train_dataset, eval_dataset, batch_size, num_workers=4):
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_dataloader, eval_dataloader

# New: Learning Rate Finder
def find_learning_rate(model, train_dataset, device, batch_size=8):
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = Adam(model.parameters(), lr=1e-7)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.1)
    
    model.train()
    lrs, losses = [], []
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        lrs.append(optimizer.param_groups[0]['lr'])
        losses.append(loss.item())
        
        if optimizer.param_groups[0]['lr'] > 10:
            break
    
    return lrs, losses

# Use this function before training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lrs, losses = find_learning_rate(model, train_dataset, device)
optimal_lr = lrs[losses.index(min(losses))]
print(f"Optimal learning rate: {optimal_lr}")

# Update the learning rate in training_args
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    eval_strategy="steps",
    eval_steps=250,
    logging_steps=50,
    learning_rate=optimal_lr,
    weight_decay=0.01,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    save_strategy="steps",
    save_steps=250,
    save_total_limit=2,
    load_best_model_at_end=True,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    fp16=True,
)

print("Before trainer initialization:")
print_gpu_memory()

# New: Custom callback for GPU monitoring
class GPUMemoryCallback(TrainerCallback):
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.progress_bar = None

    def on_train_begin(self, args, state, control, **kwargs):
        total_steps = state.max_steps if state.max_steps > 0 else args.num_train_epochs * state.num_train_epochs
        self.progress_bar = tqdm(total=total_steps, desc="Training")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:  # Log every 100 steps
            gpu = GPUtil.getGPUs()[self.gpu_id]
            memory_used = gpu.memoryUsed
            memory_total = gpu.memoryTotal
            cpu_percent = psutil.cpu_percent()
            self.progress_bar.set_postfix({
                'GPU Mem': f"{memory_used}/{memory_total} MB",
                'CPU': f"{cpu_percent}%"
            })
        self.progress_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar:
            self.progress_bar.close()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            self.progress_bar.set_postfix(logs)

# Initialize Trainer with the new callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[GPUMemoryCallback()],
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
