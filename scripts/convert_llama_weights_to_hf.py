import os
import json
import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

def convert_llama_weights_to_hf(llama_model_path, output_dir):
    # Load the model parameters
    with open(os.path.join(llama_model_path, 'params.json'), 'r') as f:
        params = json.load(f)
    
    # Print the keys in params for debugging
    print("Keys in params:", params.keys())
    
    # Create the config
    config = LlamaConfig(
        hidden_size=params['dim'],
        intermediate_size=params.get('hidden_dim', 4 * params['dim']),  # Default to 4x hidden_size if not present
        num_attention_heads=params['n_heads'],
        num_hidden_layers=params['n_layers'],
        rms_norm_eps=params.get('norm_eps', 1e-6),  # Provide a default value
        max_position_embeddings=params.get('max_seq_len', 2048),  # Provide a default value
        vocab_size=params.get('vocab_size', 32000),  # Provide a default value
    )
    
    # Create the model
    model = LlamaForCausalLM(config)
    
    # Load the weights
    state_dict = torch.load(os.path.join(llama_model_path, 'consolidated.00.pth'), map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    
    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
    tokenizer.save_pretrained(output_dir)

    print(f"Model and tokenizer saved to {output_dir}")

if __name__ == "__main__":
    llama_model_path = "/home/ubuntu/.llama/checkpoints/Llama3.2-11B-Vision-Instruct"
    output_dir = "/home/ubuntu/.llama/checkpoints/Llama3.2-11B-Vision-Instruct-HF"
    
    convert_llama_weights_to_hf(llama_model_path, output_dir)
