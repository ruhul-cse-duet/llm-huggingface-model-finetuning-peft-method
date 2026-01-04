"""
Enhanced LoRA Model Implementation
===================================

Complete LoRA implementation with support for:
- Auto-detection of target modules
- Quantization support (8-bit/4-bit)
- Model merging and unloading
- Comprehensive logging
"""

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from typing import Optional, List, Union
import logging

logger = logging.getLogger(__name__)


class LoRAModel:
    """
    LoRA Model Wrapper
    
    Example Usage:
    ```python
    model = LoRAModel.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"]
    )
    ```
    """
    
    def __init__(
        self,        base_model,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        bias: str = "none",
        task_type: str = "CAUSAL_LM",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        self.base_model = base_model
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or self._auto_detect_target_modules()
        
        # Create LoRA config
        self.config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=self.target_modules,
            bias=bias,
            task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else TaskType[task_type],
        )
        
        # Prepare for quantized training if needed
        if load_in_8bit or load_in_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, self.config)
        logger.info(f"LoRA applied with r={r}, alpha={lora_alpha}")
        self.print_trainable_parameters()
    
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs
    ):
        """
        Load pretrained model and apply LoRA
        
        Args:
            model_name: HuggingFace model identifier
            r: LoRA rank (4-64, default: 8)
            lora_alpha: Scaling factor (default: 16)
            lora_dropout: Dropout probability
            target_modules: Layers to apply LoRA
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
        """
        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            **kwargs
        )
        
        return cls(
            base_model=base_model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )
    
    def _auto_detect_target_modules(self) -> List[str]:
        """Auto-detect target modules based on model architecture"""
        model_type = self.base_model.config.model_type.lower()
        
        target_modules_map = {
            "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "gpt2": ["c_attn", "c_proj"],
            "gptj": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "gpt_neox": ["query_key_value"],
            "opt": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "bloom": ["query_key_value"],
            "falcon": ["query_key_value"],
            "mpt": ["Wqkv"],
        }
        
        for key, modules in target_modules_map.items():
            if key in model_type:
                logger.info(f"Detected {key}, using: {modules}")
                return modules
        
        logger.warning(f"Unknown model type: {model_type}")
        return ["q_proj", "v_proj"]
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_percentage = 100 * trainable_params / all_params
        
        print(f"\n{'='*60}")
        print(f"  LoRA Configuration")
        print(f"{'='*60}")
        print(f"  Trainable params: {trainable_params:,} ({trainable_percentage:.2f}%)")
        print(f"  All params: {all_params:,}")
        print(f"  LoRA rank (r): {self.r}")
        print(f"  LoRA alpha: {self.lora_alpha}")
        print(f"  Target modules: {self.target_modules}")
        print(f"{'='*60}\n")
    
    def merge_and_unload(self):
        """Merge LoRA weights into base model for inference"""
        logger.info("Merging LoRA weights into base model...")
        self.model = self.model.merge_and_unload()
        return self.model
    
    def save_pretrained(self, save_directory: str):
        """Save LoRA adapter weights"""
        self.model.save_pretrained(save_directory)
        logger.info(f"LoRA adapter saved to {save_directory}")
    
    def get_model(self):
        """Get the wrapped model"""
        return self.model
