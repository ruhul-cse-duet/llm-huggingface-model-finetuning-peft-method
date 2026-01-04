"""
Inference Script
================

Run inference with fine-tuned PEFT models.

Usage:
    python scripts/inference.py --model_path outputs/final_lora_model --prompt "Hello, how are you?"
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7):
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            num_return_sequences=1
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def main():
    parser = argparse.ArgumentParser(description="Run inference with PEFT model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to PEFT adapter")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name (auto-detected if not provided)")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Input prompt")
    parser.add_argument("--max_length", type=int, default=100,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model_path}...")
    
    # Load base model and tokenizer
    if args.base_model:
        base_model_name = args.base_model
    else:
        # Try to detect from adapter config
        import json
        with open(f"{args.model_path}/adapter_config.json", 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "gpt2")
    
    print(f"Base model: {base_model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    
    # Load PEFT adapter
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    
    print(f"\nPrompt: {args.prompt}")
    print("-" * 50)
    
    # Generate
    output = generate_text(
        model, tokenizer, args.prompt,
        max_length=args.max_length,
        temperature=args.temperature
    )
    
    print(f"Generated: {output}")
    print("-" * 50)


if __name__ == "__main__":
    main()
