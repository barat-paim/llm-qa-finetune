import os
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./llama_3_2_1b_model"

def verify_path(path):
    print(f"Checking path: {path}")
    print("Contents:")
    for item in os.listdir(path):
        print(f"  - {item}")

def verify_model_and_tokenizer(path):
    print("\nAttempting to load tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path)
        print("Tokenizer loaded successfully!")
    except Exception as e:
        print(f"Error loading tokenizer: {str(e)}")
    
    print("\nAttempting to load model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    verify_path(model_path)
    verify_model_and_tokenizer(model_path)