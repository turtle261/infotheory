#!/usr/bin/env python3
"""
Direct PyTorch to safetensors conversion for RWKV7 model
"""

import torch
import json
from safetensors.torch import save_file
import argparse

def convert_rwkv7_name(name: str) -> str:
    """Convert RWKV7 parameter names from Python format to HuggingFace format"""
    converted = name
    
    # Convert embeddings
    if converted == "emb.weight":
        converted = "model.embeddings.weight"
    
    # Convert head
    elif converted == "head.weight":
        converted = "lm_head.weight"
    
    # Convert final layer norm
    elif converted == "ln_out.weight":
        converted = "model.norm.weight"
    elif converted == "ln_out.bias":
        converted = "model.norm.bias"
    
    # Convert block parameters
    elif converted.startswith("blocks."):
        # Extract layer number
        parts = converted.split('.')
        if len(parts) >= 3:
            layer_num = parts[1]
            remaining = '.'.join(parts[2:])
            
            # Convert pre_ln (only in layer 0)
            if remaining == "ln0.weight":
                converted = f"model.layers.{layer_num}.pre_norm.weight"
            elif remaining == "ln0.bias":
                converted = f"model.layers.{layer_num}.pre_norm.bias"
            
            # Convert attention layer norm
            elif remaining == "ln1.weight":
                converted = f"model.layers.{layer_num}.attn_norm.weight"
            elif remaining == "ln1.bias":
                converted = f"model.layers.{layer_num}.attn_norm.bias"
            
            # Convert ffn layer norm
            elif remaining == "ln2.weight":
                converted = f"model.layers.{layer_num}.ffn_norm.weight"
            elif remaining == "ln2.bias":
                converted = f"model.layers.{layer_num}.ffn_norm.bias"
            
            # Convert attention parameters
            elif remaining.startswith("att."):
                att_param = remaining[4:]  # Remove "att."
                
                # Time mixing parameters
                if att_param.startswith("x_"):
                    converted = f"model.layers.{layer_num}.attn.{att_param}"
                
                # Linear projections
                elif att_param == "receptance.weight":
                    converted = f"model.layers.{layer_num}.attn.r_proj.weight"
                elif att_param == "key.weight":
                    converted = f"model.layers.{layer_num}.attn.k_proj.weight"
                elif att_param == "value.weight":
                    converted = f"model.layers.{layer_num}.attn.v_proj.weight"
                elif att_param == "output.weight":
                    converted = f"model.layers.{layer_num}.attn.o_proj.weight"
                
                # Group norm
                elif att_param == "ln_x.weight":
                    converted = f"model.layers.{layer_num}.attn.g_norm.weight"
                elif att_param == "ln_x.bias":
                    converted = f"model.layers.{layer_num}.attn.g_norm.bias"
                
                # LoRA parameters for w (decay) - w1 needs transpose, w2 stays as-is
                elif att_param == "w1":
                    converted = f"model.layers.{layer_num}.attn.w_lora.lora.0.weight"
                elif att_param == "w2":
                    converted = f"model.layers.{layer_num}.attn.w_lora.lora.2.weight"
                elif att_param == "w0":
                    converted = f"model.layers.{layer_num}.attn.w_lora.lora.2.bias"
                
                # LoRA parameters for a - a1 needs transpose, a2 stays as-is
                elif att_param == "a1":
                    converted = f"model.layers.{layer_num}.attn.a_lora.lora.0.weight"
                elif att_param == "a2":
                    converted = f"model.layers.{layer_num}.attn.a_lora.lora.2.weight"
                elif att_param == "a0":
                    converted = f"model.layers.{layer_num}.attn.a_lora.lora.2.bias"
                
                # LoRA parameters for v (only layer 1+) - v1 needs transpose, v2 stays as-is
                elif att_param == "v1":
                    converted = f"model.layers.{layer_num}.attn.v_lora.lora.0.weight"
                elif att_param == "v2":
                    converted = f"model.layers.{layer_num}.attn.v_lora.lora.2.weight"
                elif att_param == "v0":
                    converted = f"model.layers.{layer_num}.attn.v_lora.lora.2.bias"
                
                # LoRA parameters for g (gate) - g1 needs transpose, g2 stays as-is
                elif att_param == "g1":
                    converted = f"model.layers.{layer_num}.attn.g_lora.lora.0.weight"
                elif att_param == "g2":
                    converted = f"model.layers.{layer_num}.attn.g_lora.lora.2.weight"
                
                # Key scaling parameters
                elif att_param in ["k_k", "k_a", "r_k"]:
                    converted = f"model.layers.{layer_num}.attn.{att_param}"
                
                # If no match found, keep the attention parameter as-is with model prefix
                else:
                    converted = f"model.layers.{layer_num}.attn.{att_param}"
            
            # Convert FFN parameters
            elif remaining.startswith("ffn."):
                ffn_param = remaining[4:]  # Remove "ffn."
                
                if ffn_param == "x_k":
                    converted = f"model.layers.{layer_num}.ffn.x_k"
                elif ffn_param == "key.weight":
                    converted = f"model.layers.{layer_num}.ffn.key.weight"
                elif ffn_param == "value.weight":
                    converted = f"model.layers.{layer_num}.ffn.value.weight"
                else:
                    converted = f"model.layers.{layer_num}.ffn.{ffn_param}"
            
            # If no specific conversion found, add model prefix
            else:
                converted = f"model.layers.{layer_num}.{remaining}"
    
    return converted

def main():
    parser = argparse.ArgumentParser(description='Convert RWKV7 .pth to safetensors')
    parser.add_argument('--src', required=True, help='Source .pth file')
    parser.add_argument('--dest', required=True, help='Destination .safetensors file')
    parser.add_argument('--config', help='Config file to create')
    args = parser.parse_args()
    
    print(f"Converting {args.src} to {args.dest}")
    
    # Load the PyTorch model
    print("Loading PyTorch model...")
    weights = torch.load(args.src, map_location='cpu', weights_only=True)
    print(f"Loaded {len(weights)} tensors")
    
    # Convert to our naming convention
    converted_weights = {}
    for original_name, tensor in weights.items():
        converted_name = convert_rwkv7_name(original_name)
        
        # Convert to float32 and squeeze extra dimensions
        tensor = tensor.squeeze().float()
        
        # Transpose ALL LoRA weights because our Rust implementation transposes them for use
        # PyTorch: w1=[768,64], w2=[64,768]; Our Rust expects: w1_w=[64,768], w2_w=[768,64]
        if (original_name.endswith('.w1') or original_name.endswith('.w2') or 
            original_name.endswith('.a1') or original_name.endswith('.a2') or
            original_name.endswith('.v1') or original_name.endswith('.v2') or 
            original_name.endswith('.g1') or original_name.endswith('.g2')):
            tensor = tensor.t().contiguous()  # Make contiguous after transpose
            print(f"  {original_name} -> {converted_name}: {list(tensor.shape)} {tensor.dtype} (transposed)")
        else:
            print(f"  {original_name} -> {converted_name}: {list(tensor.shape)} {tensor.dtype}")
        
        converted_weights[converted_name] = tensor
    
    # Save as safetensors
    print(f"Saving {len(converted_weights)} tensors to {args.dest}")
    save_file(converted_weights, args.dest)
    
    # Create config if requested
    if args.config:
        config = {
            "vocab_size": 65536,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "head_dim": 64,
            "intermediate_size": 3072,
            "a_low_rank_dim": 64,
            "decay_low_rank_dim": 64,
            "v_low_rank_dim": 32,
            "gate_low_rank_dim": 128,
            "norm_eps": 1e-5,
        }
        
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created config file: {args.config}")
    
    print("Conversion completed successfully!")

if __name__ == "__main__":
    main()
