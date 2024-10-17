import torch
import gc

def print_gpu_memory():
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Allocated GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached GPU Memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    print(f"Free GPU Memory: {torch.cuda.memory_reserved() - torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("Initial GPU Memory State:")
print_gpu_memory()

# Try to allocate a large tensor
print("\nAttempting to allocate a 4GB tensor...")
try:
    large_tensor = torch.zeros(1024 * 1024 * 1024, device='cuda')  # 4GB tensor
    print("Allocation successful!")
    print_gpu_memory()
    del large_tensor
    torch.cuda.empty_cache()
    gc.collect()
except RuntimeError as e:
    print(f"Allocation failed: {e}")

print("\nFinal GPU Memory State:")
print_gpu_memory()

# Try to load a model
print("\nAttempting to load Llama model...")
try:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("./llama_3_2_1b_model", device_map="auto")
    print("Model loaded successfully!")
    print_gpu_memory()
    del model
    torch.cuda.empty_cache()
    gc.collect()
except Exception as e:
    print(f"Model loading failed: {e}")

print("\nFinal GPU Memory State:")
print_gpu_memory()