"""
Main Training Script
====================

Train LLM models with various PEFT techniques.

Usage:
    python scripts/train.py --config configs/lora_config.yaml
    python scripts/train.py --technique qlora --model meta-llama/Llama-2-7b-hf
"""

import argparse
import yaml
import torch
from transformers import AutoTokenizer, TrainingArguments
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.peft_techniques.lora import LoRAModel, LoRAConfig
from src.peft_techniques.qlora import QLoRAModel, QLoRAConfig
from src.peft_techniques.qlora.prefix_tuning import PrefixTuningModel, PrefixTuningConfig
from src.peft_techniques.qlora.prompt_tuning import PromptTuningModel, PromptTuningConfig
from src.peft_techniques.adapter import AdapterModel, AdapterConfig
from src.peft_techniques.ia3 import IA3Model, IA3Config
from src.peft_techniques.p_tuning import PTuningModel, PTuningConfig
from src.trainers.peft_trainer import PEFTTrainer
from src.data_loaders.dataset import TextDataset
from src.utils.helpers import setup_logging, save_config, print_trainable_parameters


TECHNIQUE_MAP = {
    "lora": (LoRAModel, LoRAConfig),
    "qlora": (QLoRAModel, QLoRAConfig),
    "prefix": (PrefixTuningModel, PrefixTuningConfig),
    "prompt": (PromptTuningModel, PromptTuningConfig),
    "adapter": (AdapterModel, AdapterConfig),
    "ia3": (IA3Model, IA3Config),
    "p_tuning": (PTuningModel, PTuningConfig)
}


def load_config(config_path: str):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train LLM with PEFT")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--technique", type=str, choices=list(TECHNIQUE_MAP.keys()),
                       help="PEFT technique to use")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Base model name or path")
    parser.add_argument("--train_data", type=str, default="data/train.json",
                       help="Training data path")
    parser.add_argument("--output_dir", type=str, default="outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting PEFT training...")
    
    # Load config
    if args.config:
        config_dict = load_config(args.config)
        technique = config_dict.get("technique", "lora")
        model_name = config_dict.get("model_name", args.model)
    else:
        technique = args.technique or "lora"
        model_name = args.model
        config_dict = {}
    
    logger.info(f"Using technique: {technique}")
    logger.info(f"Base model: {model_name}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Get PEFT model class and config
    ModelClass, ConfigClass = TECHNIQUE_MAP[technique]
    
    # Create PEFT config
    peft_config_dict = config_dict.get("peft_config", {})
    peft_config = ConfigClass(**peft_config_dict)
    
    logger.info(f"Loading model with {technique}...")
    model = ModelClass.from_pretrained(model_name, peft_config)
    
    # Print trainable parameters
    print_trainable_parameters(model.get_model())
    
    # Load dataset
    logger.info("Loading dataset...")
    train_dataset = TextDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=config_dict.get("max_length", 512)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=config_dict.get("epochs", 3),
        per_device_train_batch_size=config_dict.get("batch_size", 4),
        gradient_accumulation_steps=config_dict.get("grad_accum", 4),
        learning_rate=config_dict.get("learning_rate", 2e-4),
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=500,
        report_to="none"
    )
    
    # Create trainer
    trainer = PEFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        training_args=training_args
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    output_path = f"{args.output_dir}/final_{technique}_model"
    model.save_pretrained(output_path)
    logger.info(f"Model saved to {output_path}")
    
    # Save config
    save_config(config_dict, args.output_dir)
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
