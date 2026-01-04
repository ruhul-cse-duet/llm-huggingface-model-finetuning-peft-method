"""Prompt Tuning Model Implementation"""
from transformers import PreTrainedModel
from peft import get_peft_model, PromptTuningConfig as HFPromptConfig, PromptTuningInit
from .prompt_tuning_config import PromptTuningConfig


class PromptTuningModel:
    """Prompt Tuning model wrapper."""
    
    def __init__(self, model, config: PromptTuningConfig):
        self.model = model
        self.config = config
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, config: PromptTuningConfig, **kwargs):
        from transformers import AutoModelForCausalLM
        
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        
        # Map init method
        init_method = PromptTuningInit.RANDOM if config.prompt_tuning_init == "RANDOM" else PromptTuningInit.TEXT
        
        peft_config = HFPromptConfig(
            num_virtual_tokens=config.num_virtual_tokens,
            prompt_tuning_init=init_method,
            prompt_tuning_init_text=config.prompt_tuning_init_text,
            tokenizer_name_or_path=config.tokenizer_name_or_path or model_name_or_path,
            task_type=config.task_type
        )
        
        model = get_peft_model(base_model, peft_config)
        print(f"✓ Prompt Tuning applied with {config.num_virtual_tokens} virtual tokens")
        
        return cls(model, config)
    
    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)
    
    def get_model(self):
        return self.model
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
