"""Adapter Configuration"""
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class AdapterConfig:
    """
    Configuration for Adapter layers.
    
    Parameters:
    -----------
    adapter_dim : int
        Bottleneck dimension for adapter. Typical: 64-256
        Smaller = fewer parameters, less expressiveness
        
    target_modules : List[str]
        Which modules to add adapters to
        
    adapter_dropout : float
        Dropout probability in adapter layers
        
    init_weights : str
        Weight initialization: "bert" or "lora"
        
    scaling : float
        Scaling factor for adapter output
        
    task_type : str
        Task type for the model
    """
    
    adapter_dim: int = 64
    target_modules: Optional[List[str]] = None
    adapter_dropout: float = 0.0
    init_weights: str = "bert"
    scaling: float = 1.0
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    
    def __post_init__(self):
        if self.adapter_dim <= 0:
            raise ValueError(f"adapter_dim must be positive, got {self.adapter_dim}")
