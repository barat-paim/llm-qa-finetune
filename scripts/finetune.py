import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    TrainingArguments
)
from datasets import load_from_disk
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from torch.utils.data import Dataset
import time
from tqdm import tqdm

class SQuADDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        input_text = f"Question: {example['question']}\nContext: {example['context']}\nAnswer:"
        target_text = f" {example['answers']['text'][0]}" if example['answers']['text'] else ""

        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": targets.input_ids.squeeze()
        }

def main():
    # Load small subset of SQuAD
    print("Loading dataset...")
    squad_dataset = load_from_disk("./data/squad")
    train_data = squad_dataset['train'].select(range(100))
    eval_data = squad_dataset['validation'].select(range(20))
    
    # Load smallest LLaMA model with 8-bit quantization
    print("Loading model...")
    model_path = "./llama_3_2_1b_model"  # Adjust to your smallest LLaMA path
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    
    # Setup LoRA
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=8,  # Reduced from 16
        lora_alpha=16,  # Reduced from 32
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # Prepare datasets
    train_dataset = SQuADDataset(train_data, tokenizer)
    eval_dataset = SQuADDataset(eval_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        eval_steps=10,
        logging_steps=5,
        learning_rate=1e-4,
        fp16=True,
        optim="paged_adamw_32bit",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
    )
    
    # Custom trainer with metrics
    class MetricsTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.start_time = time.time()
            
        def log(self, logs):
            logs = logs.copy()
            # Add memory usage
            if torch.cuda.is_available():
                logs["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1e9
            
            # Add training speed
            if self.state.global_step > 0:
                elapsed_time = time.time() - self.start_time
                logs["training_samples_per_second"] = self.state.global_step * self.args.per_device_train_batch_size / elapsed_time
            
            super().log(logs)
    
    # Initialize trainer
    trainer = MetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train and track metrics
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    model.save_pretrained("./fine_tuned_llama_squad_small")
    tokenizer.save_pretrained("./fine_tuned_llama_squad_small")

if __name__ == "__main__":
    main()