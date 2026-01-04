# Copy the full content from the artifact above
# This is the complete PEFT project generator
import os
from pathlib import Path

BASE_PATH = r"E:\Data Science\ML_and_DL_project\NLP Project\LLM model finetune peft Project"

# Create all technique files directly here
def create_prefix_tuning():
    base = Path(BASE_PATH) / "src" / "peft_techniques" / "prefix_tuning"
    base.mkdir(parents=True, exist_ok=True)
    
    # __init__.py
    with open(base / "__init__.py", 'w', encoding='utf-8') as f:
        f.write('''"""
Prefix Tuning - Prepend trainable continuous vectors to transformer layers

Key Innovation:
- Adds trainable "prefix" parameters to keys and values of each attention layer  
- Prefix length typically 10-200 tokens
- Reparameterization through MLP for stability

Benefits:
- More expressive than prompt tuning (multi-layer)
- Stable training through prefix projection
- Task-specific prefixes can be cached
- ~0.1% trainable parameters

Best Use Cases:
- Table-to-text generation
- Summarization tasks
- When you need generation quality

References:
- Paper: https://arxiv.org/abs/2101.00190
"""

from .prefix_model import PrefixTuningModel
from .prefix_config import PrefixTuningConfig

__all__ = ["PrefixTuningModel", "PrefixTuningConfig"]
''')