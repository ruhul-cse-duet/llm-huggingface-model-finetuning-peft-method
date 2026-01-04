"""
Utility Functions
=================

Helper functions for logging, metrics, and general utilities.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any


def setup_logging(log_dir: str = "logs"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_config(config: Dict[str, Any], output_dir: str):
    """Save configuration to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Config saved to {config_path}")


def print_trainable_parameters(model):
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_params = 0
    
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    percentage = 100 * trainable_params / all_params
    
    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {all_params:,} || "
          f"Trainable%: {percentage:.4f}%")
    
    return trainable_params, all_params, percentage
