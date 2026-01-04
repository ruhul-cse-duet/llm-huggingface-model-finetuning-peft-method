"""
LoRA (Low-Rank Adaptation) Module
==================================

LoRA is the most popular PEFT technique that adds trainable rank decomposition 
matrices to existing weights in the model.

Key Concepts:
- Adds trainable matrices B and A to pre-trained weights W
- W' = W + BA where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
- Only trains B and A, keeping W frozen
- Reduces trainable parameters by ~10,000x
- No inference latency when matrices are merged

Benefits:
- Memory efficient: Can fine-tune 7B models with 8GB VRAM
- Fast training: Only ~1% parameters to update
- Task switching: Multiple LoRA adapters for different tasks
- Quality: Often matches full fine-tuning performance

References:
- Paper: https://arxiv.org/abs/2106.09685
- Original: "LoRA: Low-Rank Adaptation of Large Language Models"
"""

from .lora_model import LoRAModel
from .lora_config import LoRAConfig

__all__ = ["LoRAModel", "LoRAConfig"]
