"""P-Tuning v2 Configuration"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PTuningConfig:
    """
    Configuration for P-Tuning v2.
    
    Parameters:
    -----------
    num_virtual_tokens : int
        Number of virtual tokens per layer. Typical: 10-100
        
    encoder_reparameterization_type : str
        Type of encoder for prompt initialization:
        - "MLP": Use MLP encoder
        - "LSTM": Use LSTM encoder (original P-Tuning)
        
    encoder_hidden_size : int
        Hidden size of prompt encoder
        
    encoder_num_layers : int
        Number of layers in prompt encoder
        
    encoder_dropout : float
        Dropout in prompt encoder
        
    task_type : str
        Task type: typically "SEQ_CLS" or "TOKEN_CLS" for NLU
    """
    
    num_virtual_tokens: int = 20
    encoder_reparameterization_type: str = "MLP"
    encoder_hidden_size: Optional[int] = None
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.0
    task_type: str = "SEQ_CLS"
    inference_mode: bool = False
    
    def __post_init__(self):
        if self.num_virtual_tokens <= 0:
            raise ValueError(f"num_virtual_tokens must be positive")
