import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_llama_squad"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the SQuAD dataset
squad_dataset = load_dataset("squad", split="validation")

def generate_answer(question, context, max_length=50):
    input_text = f"Question: {question}\nContext: {context}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    outputs = model.generate(
        **inputs,
        max_length=len(inputs["input_ids"][0]) + max_length,
        num_return_sequences=1,
        temperature=0.7,
    )
    
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_answer.split("Answer:")[-1].strip()

# Test the model on a few examples
for i in range(5):  # Test on 5 examples
    example = squad_dataset[i]
    question = example['question']
    context = example['context']
    true_answer = example['answers']['text'][0]
    
    generated_answer = generate_answer(question, context)
    
    print(f"Question: {question}")
    print(f"Context: {context[:100]}...")  # Print first 100 characters of context
    print(f"True Answer: {true_answer}")
    print(f"Generated Answer: {generated_answer}")
    print("\n" + "="*50 + "\n")
