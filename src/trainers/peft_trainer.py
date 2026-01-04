"""
PEFT Trainer
============

Specialized trainer for PEFT methods with optimizations.
"""

from transformers import Trainer, DataCollatorForLanguageModeling
from .base_trainer import BaseTrainer


class PEFTTrainer(BaseTrainer):
    """
    Trainer optimized for PEFT fine-tuning.
    
    Features:
    - Gradient checkpointing support
    - Mixed precision training
    - PEFT-specific optimizations
    """
    
    def setup_trainer(self) -> Trainer:
        """Setup trainer with PEFT optimizations."""
        
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False  # Causal LM (not masked LM)
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model.get_model() if hasattr(self.model, 'get_model') else self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        return trainer
