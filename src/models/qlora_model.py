"""
QLoRA (Quantized LoRA) Model Implementation
LoRA with 4-bit quantization for extreme memory efficiency
"""

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from .base_model import BaseModel
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


class QLoRAModel(BaseModel):
    """
    QLoRA Model - LoRA with 4-bit Quantization
    
    Key Features:
    - 4-bit NormalFloat (NF4) quantization
    - Double quantization for constants
    - Paged optimizers for memory spikes
    - Can fine-tune 65B models on 48GB GPU
    
    Benefits:
    - 4x memory reduction compared to LoRA
    - Enables large model fine-tuning on consumer GPUs
    - Minimal performance degradation
    """
    
    def __init__(
        self,
        model_name: str,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        **kwargs
    ):
        """
        Initialize QLoRA model
        
        Args:
            model_name: Base model identifier
            lora_r: LoRA rank (higher for QLoRA, typically 64)
            lora_alpha: LoRA alpha scaling
            lora_dropout: Dropout probability
            target_modules: Modules to apply LoRA
            bias: Bias training strategy
            **kwargs: Additional arguments
        """
        # Force 4-bit quantization
        kwargs['load_in_4bit'] = True
        kwargs['load_in_8bit'] = False
        
        # Initialize base model with 4-bit quantization
        super().__init__(model_name, **kwargs)
        
        # Auto-detect target modules if not provided
        if target_modules is None:
            target_modules = self._get_default_target_modules()
        
        # QLoRA configuration
        self.peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Prepare model for k-bit training
        logger.info("Preparing model for 4-bit training")
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True
        )
        
        # Apply QLoRA
        logger.info("Applying QLoRA to model")
        self.model = get_peft_model(self.model, self.peft_config)
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def _get_default_target_modules(self) -> List[str]:
        """Get default target modules"""
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
