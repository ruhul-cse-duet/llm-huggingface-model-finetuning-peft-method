"""
Prefix Tuning Configuration
============================
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class PrefixTuningConfig:
    """
    Configuration for Prefix Tuning.
    
    Parameters:
    -----------
    num_virtual_tokens : int
        Number of virtual tokens (prefix length).
        Typical values: 10-100
        More tokens = more expressiveness but more parameters
        
    encoder_hidden_size : Optional[int]
        Hidden size of the prefix encoder (if using)
        If None, prefix parameters are optimized directly
        
    prefix_projection : bool
        Whether to use MLP reparameterization for prefix.
        True: P = MLP(P_θ) - more stable training
        False: Optimize P directly - simpler but less stable
        
    task_type : str
        Type of task: "CAUSAL_LM", "SEQ_2_SEQ_LM"
        
    inference_mode : bool
        Whether in inference mode
    
    Examples:
    ---------
    >>> config = PrefixTuningConfig(
    ...     num_virtual_tokens=20,
    ...     prefix_projection=True,
    ...     encoder_hidden_size=512
    ... )
    """
    
    num_virtual_tokens: int = 20
    encoder_hidden_size: Optional[int] = None
    prefix_projection: bool = False
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    token_dim: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_layers: Optional[int] = None
    
    def __post_init__(self):
        if self.num_virtual_tokens <= 0:
            raise ValueError(f"num_virtual_tokens must be positive, got {self.num_virtual_tokens}")
