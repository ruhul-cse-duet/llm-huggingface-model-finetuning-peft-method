"""
Evaluation Script
=================

Evaluate fine-tuned PEFT models.

Usage:
    python scripts/evaluate.py --model_path outputs/final_lora_model --test_data data/test.json
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from torch.utils.data import DataLoader
import sys
from pathlib import Path
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from src.data_loaders.dataset import TextDataset


def evaluate_perplexity(model, dataloader, device):
    """Calculate perplexity on test set."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item() * input_ids.size(0)
            total_tokens += input_ids.size(0)
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    return perplexity.item(), avg_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate PEFT model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to PEFT adapter")
    parser.add_argument("--test_data", type=str, required=True,
                       help="Test data path")
    parser.add_argument("--base_model", type=str, default=None,
                       help="Base model name")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    if args.base_model:
        base_model_name = args.base_model
    else:
        with open(f"{args.model_path}/adapter_config.json", 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "gpt2")
    
    print(f"Loading model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.to(device)
    
    # Load test data
    print(f"Loading test data from {args.test_data}")
    test_dataset = TextDataset(
        data_path=args.test_data,
        tokenizer=tokenizer,
        max_length=512
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Evaluate
    print("Evaluating...")
    perplexity, avg_loss = evaluate_perplexity(model, test_loader, device)
    
    print("\n" + "="*50)
    print(f"Evaluation Results:")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Average Loss: {avg_loss:.4f}")
    print("="*50)
    
    # Save results
    results = {
        "perplexity": perplexity,
        "average_loss": avg_loss,
        "test_samples": len(test_dataset)
    }
    
    results_path = f"{args.model_path}/evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
