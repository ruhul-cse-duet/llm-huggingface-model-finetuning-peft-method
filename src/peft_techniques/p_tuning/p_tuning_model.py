"""P-Tuning v2 Model Implementation"""
from transformers import PreTrainedModel
from peft import get_peft_model, PromptEncoderConfig, PromptEncoderReparameterizationType
from .p_tuning_config import PTuningConfig


class PTuningModel:
    """P-Tuning v2 model wrapper."""
    
    def __init__(self, model, config: PTuningConfig):
        self.model = model
        self.config = config
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, config: PTuningConfig, **kwargs):
        from transformers import AutoModelForSequenceClassification
        
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, **kwargs
        )
        
        # Map reparameterization type
        reparam_type = (
            PromptEncoderReparameterizationType.MLP 
            if config.encoder_reparameterization_type == "MLP" 
            else PromptEncoderReparameterizationType.LSTM
        )
        
        peft_config = PromptEncoderConfig(
            num_virtual_tokens=config.num_virtual_tokens,
            encoder_reparameterization_type=reparam_type,
            encoder_hidden_size=config.encoder_hidden_size,
            encoder_num_layers=config.encoder_num_layers,
            encoder_dropout=config.encoder_dropout,
            task_type=config.task_type
        )
        
        model = get_peft_model(base_model, peft_config)
        print(f"✓ P-Tuning v2 applied with {config.num_virtual_tokens} tokens per layer")
        
        return cls(model, config)
    
    def save_pretrained(self, save_path: str):
        self.model.save_pretrained(save_path)
    
    def get_model(self):
        return self.model
    
    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
