"""
QLoRA (Quantized LoRA) Module
==============================

QLoRA is an advanced version of LoRA that uses 4-bit quantization to dramatically
reduce memory requirements while maintaining model quality.

Key Innovations:
1. 4-bit NormalFloat (NF4) quantization - optimal for normally distributed weights
2. Double quantization - quantizes the quantization constants themselves
3. Paged optimizers - handles memory spikes during gradient computation

Mathematical Foundation:
- Base model quantized to 4-bit: W_4bit = Quantize_4bit(W_fp16)
- LoRA applied on top: W' = W_4bit + BA (B, A in bfloat16)
- Memory savings: 75% compared to standard LoRA

Benefits:
- Train 65B models on single 24GB GPU (vs 4x 80GB without QLoRA)
- Quality loss < 1% compared to full 16-bit fine-tuning
- Perfect for resource-constrained environments
- Enables democratized LLM fine-tuning

When to use QLoRA:
✓ Large models (13B+) with limited GPU memory
✓ Need to train on consumer hardware
✓ Want to fine-tune multiple large models

When to use LoRA instead:
✓ Smaller models (7B-)
✓ Have sufficient GPU memory
✓ Need absolute maximum quality

References:
- Paper: https://arxiv.org/abs/2305.14314
- "QLoRA: Efficient Finetuning of Quantized LLMs"
"""

from .qlora_model import QLoRAModel
from .qlora_config import QLoRAConfig

__all__ = ["QLoRAModel", "QLoRAConfig"]
