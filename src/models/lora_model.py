"""
LoRA (Low-Rank Adaptation) Model Implementation
Efficient fine-tuning using low-rank decomposition matrices
"""

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from .base_model import BaseModel
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class LoRAModel(BaseModel):
    """
    LoRA Model for Parameter-Efficient Fine-tuning
    
    LoRA adds trainable rank decomposition matrices to attention layers:
    W' = W + BA where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
    
    Benefits:
    - Reduces trainable parameters by 10,000x
    - No inference latency when merged
    - Memory efficient training
    - Can be easily swapped for different tasks
    """
    
    def __init__(
        self,
        model_name: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        **kwargs
    ):
        """
        Initialize LoRA model
        
        Args:
            model_name: Base model identifier
            lora_r: LoRA rank (typically 4-64)
            lora_alpha: LoRA alpha scaling (typically 2*lora_r)
            lora_dropout: Dropout probability
            target_modules: Modules to apply LoRA (e.g., ["q_proj", "v_proj"])
            bias: Bias training strategy ("none", "all", "lora_only")
            task_type: Task type for PEFT
            **kwargs: Additional arguments for BaseModel
        """
        # Initialize base model
        super().__init__(model_name, **kwargs)
        
        # Default target modules for common architectures
        if target_modules is None:
            # Auto-detect based on model architecture
            target_modules = self._get_target_modules()
        
        # LoRA Configuration
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type,
        )
        
        # Prepare model for training if quantized
        if self.load_in_8bit or self.load_in_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Apply LoRA
        logger.info("Applying LoRA to model")
        self.model = get_peft_model(self.model, self.peft_config)
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def _get_target_modules(self) -> List[str]:
        """Auto-detect target modules based on model architecture"""
        model_type = self.model.config.model_type.lower()
        
        # Common target modules for different architectures
        target_modules_map = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "gpt2": ["c_attn"],
            "gptj": ["q_proj", "v_proj"],
            "gpt_neox": ["query_key_value"],
            "opt": ["q_proj", "v_proj"],
            "bloom": ["query_key_value"],
            "t5": ["q", "v"],
        }
        
        for key in target_modules_map:
            if key in model_type:
                logger.info(f"Detected {key} architecture, using target modules: {target_modules_map[key]}")
                return target_modules_map[key]
        
        # Default fallback
        logger.warning(f"Unknown model type {model_type}, using default target modules")
        return ["q_proj", "v_proj"]
    
    def merge_and_unload(self):
        """Merge LoRA weights with base model and unload"""
        logger.info("Merging LoRA weights with base model")
        self.model = self.model.merge_and_unload()
        return self.model
