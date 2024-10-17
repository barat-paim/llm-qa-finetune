import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# Attempt to login
token = os.environ.get("HUGGINGFACE_TOKEN")
if not token:
    raise EnvironmentError("Please set the HUGGINGFACE_TOKEN environment variable")

login(token=token)

# The model name
model_name = "meta-llama/Llama-3.2-1B"

print(f"Attempting to download {model_name}")

try:
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("Downloading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    print("Download complete!")
    
    # Optional: Save the model and tokenizer locally
    save_directory = "./llama_3_2_1b_model"
    print(f"Saving model and tokenizer to {save_directory}")
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    print("Model and tokenizer saved successfully!")

except Exception as e:
    print(f"An error occurred: {str(e)}")
