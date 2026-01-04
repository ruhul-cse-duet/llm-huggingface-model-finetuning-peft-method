"""IA³ Model Implementation"""
from transformers import PreTrainedModel
from peft import get_peft_model, IA3Config as HFIA3Config
from .ia3_config import IA3Config


class IA3Model:
    """IA³ model wrapper."""
    
    def __init__(self, model, config: IA3Config):
        self.model = model
        self.config = config
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, config: IA3Config, **kwargs):
        from transformers import AutoModelForCausalLM
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        
        peft_config = HFIA3Config(
            target_modules=config.target_modules,
            feedforward_modules=config.feedforward_modules,
            fan_in_fan_out=config.fan_in_fan_out,
            init_ia3_weights=config.init_ia3_weights,
            task_type=config.task_type,
            modules_to_save=config.modules_to_save
        )
        
        model = get_peft_model(base_model, peft_config)
        print(f"✓ IA³ applied with zero inference overhead!")
        cls._print_trainable_parameters(model)
        
        return cls(model, config)
    
    @staticmethod
    def _print_trainable_parameters(model):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} ({100*trainable/total:.4f}%) | Total: {total:,}")
    
    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)
    
    def get_model(self):
        return self.model
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
