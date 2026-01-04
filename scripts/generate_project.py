"""
Project Generator Script
========================
Generates all PEFT technique files with comprehensive documentation.
"""

import os
from pathlib import Path

BASE_PATH = Path("E:/Data Science/ML_and_DL_project/NLP Project/LLM model finetune peft Project")

# File templates for all PEFT techniques
PEFT_TECHNIQUES = {
    "prefix_tuning": {
        "description": """Prefix Tuning - Prepends trainable continuous vectors to each layer
        
Key Concepts:
- Adds trainable prefix parameters to each transformer layer
- Prefix acts like soft prompts in continuous space
- Original model weights remain frozen
- Only 0.01-1% parameters trainable

Mathematical:
h_i = Transformer_i([P_θ; h_{i-1}])
where P_θ are trainable prefix embeddings

Benefits:
- Extremely parameter efficient
- Works well for generation tasks
- Easy to switch between tasks
- No architectural changes needed

Best for:
- Table-to-text generation
- Summarization
- Data-to-text tasks
- Multi-task learning""",
        "config_params": {
            "num_virtual_tokens": 20,
            "prefix_projection": False,
        }
    },
    
    "prompt_tuning": {
        "description": """Prompt Tuning - Optimizes continuous prompts at input layer only
        
Key Concepts:
- Adds soft prompts only to input embeddings
- Most parameter-efficient method (0.001-0.01%)
- Works exceptionally well with large models (>10B)
