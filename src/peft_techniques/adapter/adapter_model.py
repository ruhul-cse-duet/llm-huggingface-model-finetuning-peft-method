"""Adapter Model Implementation"""
from transformers import PreTrainedModel
from peft import get_peft_model, AdapterConfig as HFAdapterConfig
from .adapter_config import AdapterConfig


class AdapterModel:
    """Adapter model wrapper."""
    
    def __init__(self, model, config: AdapterConfig):
        self.model = model
        self.config = config
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, config: AdapterConfig, **kwargs):
        from transformers import AutoModelForCausalLM
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        
        peft_config = HFAdapterConfig(
            adapter_hidden_size=config.adapter_dim,
            target_modules=config.target_modules,
            adapter_dropout=config.adapter_dropout,
            init_weights=config.init_weights,
            scaling=config.scaling,
            task_type=config.task_type
        )
        
        model = get_peft_model(base_model, peft_config)
        print(f"✓ Adapter layers applied with bottleneck dim={config.adapter_dim}")
        
        return cls(model, config)
    
    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)
    
    def get_model(self):
        return self.model
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
