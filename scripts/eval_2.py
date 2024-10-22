import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Define compute_f1 and compute_exact functions here if they're not in a separate utils file
def compute_f1(prediction, ground_truth):
    prediction_tokens = prediction.lower().split()
    ground_truth_tokens = ground_truth.lower().split()
    common = set(prediction_tokens) & set(ground_truth_tokens)
    if not common:
        return 0
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def compute_exact(prediction, ground_truth):
    return int(prediction.lower() == ground_truth.lower())

# Load the pre-trained LLaMA model
model_path = "./llama_3_2_1b_model"  # Path to the pre-trained model
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set padding token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate single question
def evaluate_single_question(model, tokenizer, device, question, context, true_answers):
    model.eval()
    
    with torch.no_grad():
        # Prepare input
        input_text = f"Question: {question}\nContext: {context}\nAnswer:"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
            padding_side="left"
        ).to(device)

        # Generate answer
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=model.config.eos_token_id,
            eos_token_id=model.config.eos_token_id,
            use_cache=True
        )
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True).replace("Answer:", "").strip()

        # Calculate F1 and EM
        best_f1 = max(compute_f1(generated_answer, true_ans) for true_ans in true_answers)
        best_em = max(compute_exact(generated_answer, true_ans) for true_ans in true_answers)

        return generated_answer, best_f1, best_em

# Example usage
question = "Which NFL team represented the AFC at Super Bowl 50?"
context = "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24–10 to earn their third Super Bowl title."
true_answers = ['Denver Broncos', 'Denver Broncos', 'Denver Broncos']

generated_answer, f1_score, em_score = evaluate_single_question(model, tokenizer, device, question, context, true_answers)

print(f"Question: {question}")
print(f"Context: {context}")
print(f"Generated Answer: {generated_answer}")
print(f"True Answers: {true_answers}")
print(f"F1 Score: {f1_score}")
print(f"Exact Match Score: {em_score}")
