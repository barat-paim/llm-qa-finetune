import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

def print_directory_contents(path):
    print(f"Contents of {path}:")
    for item in os.listdir(path):
        print(f"  - {item}")

def test_model_loading():
    model_path = "./llama_3_2_1b_model"  # Update this to your local model path if different
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Testing model loading from: {model_path}")
    print(f"Using device: {device}")
    print_directory_contents(model_path)

    try:
        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Tokenizer loaded successfully")

        # Load model
        print("\nLoading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
        print("Model loaded successfully")

        # Move model to device
        model.to(device)
        print(f"Model moved to {device} successfully")

        # Test inference
        print("\nTesting inference...")
        prompt = "Translate the following English text to French: 'Hello, how are you?'"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model output: {response}")

        print("\nAll tests passed successfully!")
        return True

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return False

if __name__ == "__main__":
    test_model_loading()
