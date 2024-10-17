import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set the model name and save directory
model_name = "meta-llama/Llama-3.2-1B-Instruct"
save_directory = "/home/ubuntu/.llama/checkpoints/Llama3.2-1B-Instruct"

# Download and save tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_directory)
print("Tokenizer downloaded and saved.")

# Download and save model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use half-precision to save memory
    device_map="auto",  # Automatically distribute model across available GPUs
)
model.save_pretrained(save_directory)
print("Model downloaded and saved.")

# Move model to the appropriate device
model.to(device)

# Example usage
prompt = "Explain the concept of machine learning in simple terms."
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=200)

# Decode and print the output
print("\nModel output:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Verify the download
import os
print("\nContents of the save directory:")
print(os.listdir(save_directory))
