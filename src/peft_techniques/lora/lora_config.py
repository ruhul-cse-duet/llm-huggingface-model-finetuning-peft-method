"""
LoRA Configuration Class
========================

Configuration for LoRA fine-tuning.

Parameters explained:
- r (rank): Dimensionality of the low-rank matrices (4-64)
  * Lower r = fewer parameters, less expressive
  * Higher r = more parameters, more expressive
  * Typical: 8-16 for most tasks
  
- lora_alpha: Scaling factor for LoRA weights
  * Controls the magnitude of LoRA updates
  * Typically set to 2*r or r
  * Higher values = stronger adaptation
  
- lora_dropout: Dropout probability (0.0-0.5)
  * Regularization to prevent overfitting
  * Typical: 0.05-0.1
  
- target_modules: Which layers to apply LoRA
  * ["q_proj", "v_proj"] = Query and Value attention
  * Can add "k_proj", "o_proj" for more capacity
  * More modules = more parameters but better performance
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    """LoRA configuration class"""
    
    # Core LoRA parameters
    r: int = 8  # Rank of LoRA matrices
    lora_alpha: int = 16  # Scaling parameter
    lora_dropout: float = 0.05  # Dropout probability
    
    # Target modules
    target_modules: Optional[List[str]] = None  # Auto-detect if None
    
    # Bias configuration
    bias: str = "none"  # Options: "none", "all", "lora_only"
    
    # Task type
    task_type: str = "CAUSAL_LM"  # CAUSAL_LM, SEQ_2_SEQ_LM, etc.
    
    # Advanced options
    fan_in_fan_out: bool = False  # Set True for Conv1D layers
    modules_to_save: Optional[List[str]] = None  # Additional modules to train
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.r > 0, "r must be positive"
        assert 0 <= self.lora_dropout < 1, "dropout must be in [0, 1)"
        assert self.bias in ["none", "all", "lora_only"]
