"""Prompt Tuning Configuration"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTuningConfig:
    """
    Configuration for Prompt Tuning.
    
    Parameters:
    -----------
    num_virtual_tokens : int
        Number of virtual prompt tokens. Typical: 8-100
        
    prompt_tuning_init : str
        Initialization method:
        - "RANDOM": Random initialization
        - "TEXT": Initialize from text tokens
        
    prompt_tuning_init_text : Optional[str]
        Text to initialize prompts (if using TEXT init)
        
    tokenizer_name_or_path : Optional[str]
        Tokenizer for text initialization
        
    task_type : str
        Task type: "CAUSAL_LM", "SEQ_2_SEQ_LM", "SEQ_CLS"
    """
    
    num_virtual_tokens: int = 20
    prompt_tuning_init: str = "RANDOM"
    prompt_tuning_init_text: Optional[str] = None
    tokenizer_name_or_path: Optional[str] = None
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
