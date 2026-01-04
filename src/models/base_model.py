"""
Base Model Class for PEFT Fine-tuning
Provides common functionality for all PEFT methods
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from typing import Optional, Dict, Any, Union
import logging

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all PEFT models"""
    
    def __init__(
        self,
        model_name: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        trust_remote_code: bool = False,
        use_auth_token: Optional[str] = None,
    ):
        """
        Initialize base model
        
        Args:
            model_name: HuggingFace model identifier
            load_in_8bit: Load model in 8-bit precision
            load_in_4bit: Load model in 4-bit precision
            device_map: Device mapping strategy
            torch_dtype: PyTorch dtype for model weights
            trust_remote_code: Trust remote code
            use_auth_token: HuggingFace authentication token
        """
        self.model_name = model_name
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        
        # Quantization config
        quantization_config = None
        if load_in_4bit or load_in_8bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=device_map,
            torch_dtype=torch_dtype if not quantization_config else None,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
        
        # Load tokenizer
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id
    
    def prepare_model_for_training(self):
        """Prepare model for training"""
        # Enable gradient checkpointing
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
    
    def print_trainable_parameters(self):
        """Print the number of trainable parameters"""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        percentage = 100 * trainable_params / all_param
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {percentage:.4f}%"
        )
        return trainable_params, all_param, percentage
    
    def generate(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs
    ) -> Union[str, list]:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            num_return_sequences: Number of sequences to generate
            
        Returns:
            Generated text or list of texts
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        
        if num_return_sequences == 1:
            return generated_texts[0]
        return generated_texts
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer"""
        logger.info(f"Saving model to {save_directory}")
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load model from checkpoint"""
        logger.info(f"Loading model from {model_path}")
        instance = cls(model_name=model_path, **kwargs)
        return instance
