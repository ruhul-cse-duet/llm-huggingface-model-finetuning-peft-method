"""
Base Trainer Class
==================

Base trainer for all PEFT fine-tuning methods.
"""

import os
import torch
from typing import Optional, Dict, Any
from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback


class BaseTrainer:
    """
    Base trainer class with common functionality.
    
    Handles:
    - Training loop setup
    - Checkpointing
    - Logging
    - Evaluation
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_dataset=None,
        training_args: Optional[TrainingArguments] = None,
        **kwargs
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Default training arguments if not provided
        if training_args is None:
            training_args = self._get_default_training_args()
        
        self.training_args = training_args
    
    def _get_default_training_args(self) -> TrainingArguments:
        """Get default training arguments."""
        return TrainingArguments(
            output_dir="./outputs",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=10,
            save_steps=500,
            eval_steps=500,
            save_total_limit=3,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="steps",
            load_best_model_at_end=True,
            report_to="none",
        )
    
    def setup_trainer(self) -> Trainer:
        """Setup HuggingFace Trainer."""
        raise NotImplementedError("Subclasses must implement setup_trainer")
    
    def train(self):
        """Start training."""
        trainer = self.setup_trainer()
        trainer.train()
        return trainer
    
    def evaluate(self):
        """Evaluate the model."""
        trainer = self.setup_trainer()
        results = trainer.evaluate()
        return results
    
    def save_model(self, output_dir: str):
        """Save the model."""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"✓ Model saved to {output_dir}")
