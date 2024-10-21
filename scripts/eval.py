import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import logging

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

# Create a DataLoader for batch evaluation
logger.info(f"Loaded {len(eval_dataset)} examples")
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)
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
def evaluate_model(eval_loader, model, device):
    total_f1 = 0
    total_em = 0
    num_examples = len(eval_loader.dataset)
    
    logger.info(f"Evaluating {num_examples} examples....")
    model.eval()
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for batch_idx, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            questions = batch['question']
            contexts = batch['context']
            true_answers = batch['answers']['text']
            
            logger.info(f"Evaluating batch {batch_idx + 1} of {len(eval_loader)}")
            # Prepare input text in batch
            batch_inputs = [f"Question: {q}\nContext: {c}\nAnswer:" for q, c in zip(questions, contexts)]
            inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # Generate answers
            logger.info(f"Generating answers for batch {batch_idx + 1} of {len(eval_loader)}")
            outputs = model.generate(**inputs, max_new_tokens=50)
            generated_answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Evaluate each generated answer in the batch
            for gen_answer, true_ans_list in zip(generated_answers, true_answers):
                gen_answer = gen_answer.replace("Answer:", "").strip()
                best_f1 = max(compute_f1(gen_answer, true_ans) for true_ans in true_ans_list)
                best_em = max(compute_exact(gen_answer, true_ans) for true_ans in true_ans_list)
                
                total_f1 += best_f1
                total_em += best_em
    
    avg_f1 = total_f1 / num_examples
    avg_em = total_em / num_examples
    
    logger.info(f"Finished evaluating all examples")
    return avg_f1, avg_em

# Run the evaluation and print the results
avg_f1, avg_em = evaluate_model(eval_loader, model, device)
logger.info(f"Average F1 Score: {avg_f1:.4f}")
logger.info(f"Average Exact Match (EM) Score: {avg_em:.4f}")
