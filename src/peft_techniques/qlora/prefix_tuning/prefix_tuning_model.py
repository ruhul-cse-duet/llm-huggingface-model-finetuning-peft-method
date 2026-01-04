"""
Prefix Tuning Model Implementation
===================================
"""

from typing import Optional
from transformers import PreTrainedModel
from peft import get_peft_model, PrefixTuningConfig as HFPrefixConfig
from .prefix_tuning_config import PrefixTuningConfig


class PrefixTuningModel:
    """
    Prefix Tuning model wrapper.
    
    Adds trainable prefix vectors to attention layers.
    """
    
    def __init__(self, model, config: PrefixTuningConfig):
        self.model = model
        self.config = config
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, config: PrefixTuningConfig, **kwargs):
        from transformers import AutoModelForCausalLM
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        
        peft_config = HFPrefixConfig(
            num_virtual_tokens=config.num_virtual_tokens,
            encoder_hidden_size=config.encoder_hidden_size,
            prefix_projection=config.prefix_projection,
            task_type=config.task_type,
            inference_mode=config.inference_mode
        )
        
        model = get_peft_model(base_model, peft_config)
        print(f"✓ Prefix Tuning applied with {config.num_virtual_tokens} virtual tokens")
        
        return cls(model, config)
    
    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)
    
    def get_model(self):
        return self.model
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
