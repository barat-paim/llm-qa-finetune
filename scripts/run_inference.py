import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def run_inference(model_path, tokenizer_path, input_prompt):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input prompt
    inputs = tokenizer(input_prompt, return_tensors="pt").to(device)

    # Generate predictions
    outputs = model.generate(**inputs, max_length=200)

    # Decode the output to text
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    model_path = "./llama_3_2_1b_model"
    tokenizer_path = model_path  # Tokenizer is in the same path as the model
    input_prompt = "Explain the concept of machine learning in simple terms."

    result = run_inference(model_path, tokenizer_path, input_prompt)
    print("Generated response:", result)