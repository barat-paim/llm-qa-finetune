import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import logging
from collections import Counter
import psutil
import time
import os
import GPUtil
from torch.nn.utils.rnn import pad_sequence

# Monitoring GPU
def monitor_gpu():
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming we're using the first GPU
        return f"GPU Memory Usage: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil*100:.2f}%)"
    return "No GPU detected"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the fine-tuned model
model_path = "./fine_tuned_llama_squad"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = 'left'

# Check if CUDA (GPU) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
logger.info(f"Model is on {device}")

# Load the SQuAD validation dataset
squad_dataset = load_from_disk("./data/squad")
eval_dataset = squad_dataset['validation']

# Define a custom collate function
def custom_collate_fn(batch):
    # Separate the different items in the batch
    questions = [item['question'] for item in batch]
    contexts = [item['context'] for item in batch]
    answers = [item['answers'] for item in batch]
    
    return {
        'question': questions,
        'context': contexts,
        'answers': answers
    }

# Create a DataLoader for batch evaluation
logger.info(f"Loaded {len(eval_dataset)} examples")
eval_loader = DataLoader(eval_dataset, batch_size=12, shuffle=False, collate_fn=custom_collate_fn)
logger.info(f"Loaded {len(eval_loader)} batches")

# Define functions to calculate F1 Score and Exact Match (EM)
def compute_f1(predicted, ground_truth):
    pred_tokens = predicted.split()
    truth_tokens = ground_truth.split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def compute_exact(predicted, ground_truth):
    return int(predicted.strip() == ground_truth.strip())

# Function to evaluate the model using batches
def evaluate_model(eval_loader, model, tokenizer, device):
    total_f1 = 0
    total_em = 0
    num_examples = len(eval_loader.dataset)
    incorrect_predictions = []
    
    logger.info(f"Evaluating {num_examples} examples....")
    model.eval()
    
    model.config.pad_token_id = model.config.eos_token_id
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            if batch_idx % 100 == 0:  # Log GPU usage every 100 batches
                logger.info(monitor_gpu())
            
            questions = batch['question']
            contexts = batch['context']
            true_answers = batch['answers']
            
            inputs = tokenizer(
                [f"Question: {q}\nContext: {c}\nAnswer:" for q, c in zip(questions, contexts)],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # Adjust this based on your input length
                padding_side="left"
            ).to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                pad_token_id=model.config.eos_token_id,
                eos_token_id=model.config.eos_token_id,
                use_cache=True
            )
            generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for gen_answer, true_ans, question, context in zip(generated_answers, true_answers, questions, contexts):
                gen_answer = gen_answer.replace("Answer:", "").strip()
                true_ans_list = true_ans['text']  # Access 'text' field here
                best_f1 = max(compute_f1(gen_answer, true_ans) for true_ans in true_ans_list if true_ans)
                best_em = max(compute_exact(gen_answer, true_ans) for true_ans in true_ans_list if true_ans)
                
                if best_f1 < 0.5 and best_em == 0:
                    incorrect_predictions.append({
                        "question": question,
                        "context": context,
                        "generated_answer": gen_answer,
                        "true_answers": true_ans_list
                    })
                
                total_f1 += best_f1
                total_em += best_em
    
    avg_f1 = total_f1 / num_examples
    avg_em = total_em / num_examples
    logger.info(f"Finished evaluating all examples")
    logger.info(monitor_gpu())  # Log final GPU usage
    
    # Log incorrect predictions for further analysis
    logger.info(f"Number of Incorrect Predictions: {len(incorrect_predictions)}")
    for i, failure in enumerate(incorrect_predictions[:5]):  # Limit to 5 examples for display
        logger.info(f"\nIncorrect Prediction {i+1}:")
        logger.info(f"Question: {failure['question']}")
        logger.info(f"Context: {failure['context']}")
        logger.info(f"Generated Answer: {failure['generated_answer']}")
        logger.info(f"True Answers: {failure['true_answers']}")
    
    return avg_f1, avg_em, incorrect_predictions

# Run the evaluation
avg_f1, avg_em, incorrect_predictions = evaluate_model(eval_loader, model, tokenizer, device)
logger.info(f"Average F1 Score: {avg_f1:.4f}")
logger.info(f"Average Exact Match (EM) Score: {avg_em:.4f}")
logger.info(f"Total Incorrect Predictions: {len(incorrect_predictions)}")
