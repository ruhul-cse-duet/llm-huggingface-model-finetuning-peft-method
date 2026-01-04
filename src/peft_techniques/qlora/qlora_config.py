"""
QLoRA Configuration
===================

Configuration for 4-bit quantized LoRA training.

Key Parameters Explained:

Quantization Parameters:
- load_in_4bit: Enable 4-bit quantization (True for QLoRA)
- bnb_4bit_compute_dtype: Computation dtype (float16/bfloat16)
  * bfloat16 recommended for stability
  * float16 slightly faster but less stable
  
- bnb_4bit_quant_type: Quantization type
  * "nf4": NormalFloat4, optimal for normally distributed weights
  * "fp4": Standard 4-bit float
  
- bnb_4bit_use_double_quant: Double quantization
  * Quantizes quantization constants
  * Saves additional 0.4 bits per parameter
  * Minimal quality impact

LoRA Parameters:
- Same as standard LoRA (r, alpha, dropout, target_modules)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


@dataclass
class QLoRAConfig:
    """QLoRA configuration with 4-bit quantization"""
    
    # Quantization settings
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_quant_type: str = "nf4"  # "nf4" or "fp4"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA settings (inherited)
    r: int = 16  # Higher rank for quantized models
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Advanced options
    use_gradient_checkpointing: bool = True  # Further reduce memory
    max_memory_MB: Optional[int] = None  # Per-device memory limit
    
    def __post_init__(self):
        """Validate QLoRA configuration"""
        assert self.load_in_4bit, "QLoRA requires load_in_4bit=True"
        assert self.bnb_4bit_quant_type in ["nf4", "fp4"]
        assert self.r > 0, "LoRA rank must be positive"
        
        # Convert dtype string if needed
        if isinstance(self.bnb_4bit_compute_dtype, str):
            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32
            }
            self.bnb_4bit_compute_dtype = dtype_map.get(
                self.bnb_4bit_compute_dtype, torch.bfloat16
            )
