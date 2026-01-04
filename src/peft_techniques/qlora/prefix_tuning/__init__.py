"""
Prefix Tuning PEFT Technique
=============================

Prefix Tuning prepends trainable continuous vectors (prefixes) to the key and value
tensors at every transformer layer, while keeping the model parameters frozen.

Key Concepts:
- Virtual tokens prepended to each layer
- Only prefix parameters are trainable (~0.01-1% of model)
- Works like "soft prompts" that guide the model's behavior
- Particularly effective for generation tasks

Architecture:
    For each layer l:
    K_l = concat([P_K^l, K_l])  # Prefix prepended to keys
    V_l = concat([P_V^l, V_l])  # Prefix prepended to values
    
    where P_K^l, P_V^l are trainable prefix parameters

Use Cases:
- Table-to-text generation
- Summarization
- Data-to-text tasks
- Best for: Generation tasks with structured inputs

Memory Efficiency:
- Adds ~0.1-1% trainable parameters
- Small inference overhead (prefix tokens increase sequence length)
"""

from .prefix_tuning_model import PrefixTuningModel
from .prefix_tuning_config import PrefixTuningConfig

__all__ = ['PrefixTuningModel', 'PrefixTuningConfig']
