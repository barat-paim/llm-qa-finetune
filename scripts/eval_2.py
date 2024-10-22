import torch
from utils import compute_f1, compute_exact
from transformers import AutoTokenizer, AutoModelForCausalLM


# load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-small-instruct")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-small-instruct")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load eval data
eval_data = load_jsonl("data/eval.jsonl")

# evaluate single question
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

print(f"Generated Answer: {generated_answer}")
print(f"F1 Score: {f1_score}")
print(f"Exact Match Score: {em_score}")