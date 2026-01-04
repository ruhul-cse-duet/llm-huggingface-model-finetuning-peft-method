"""IA³ Configuration"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class IA3Config:
    """
    Configuration for IA³.
    
    Parameters:
    -----------
    target_modules : List[str]
        Modules to apply IA³ rescaling to.
        Typical: ["k_proj", "v_proj", "down_proj"]
        
    feedforward_modules : List[str]
        Feed-forward modules to apply IA³ to.
        Typical: ["down_proj"] or ["fc2"]
        
    fan_in_fan_out : bool
        Set True for Conv1D layers
        
    init_ia3_weights : bool
        Whether to initialize IA³ weights (default: ones)
        
    task_type : str
        Task type for the model
    """
    
    target_modules: List[str] = None
    feedforward_modules: List[str] = None
    fan_in_fan_out: bool = False
    init_ia3_weights: bool = True
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    modules_to_save: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["k_proj", "v_proj", "down_proj"]
        if self.feedforward_modules is None:
            self.feedforward_modules = ["down_proj"]
