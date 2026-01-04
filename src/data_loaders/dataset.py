"""
Dataset Utilities
=================

Custom dataset classes for PEFT training.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import json


class TextDataset(Dataset):
    """
    Simple text dataset for language modeling.
    
    Examples:
    ---------
    >>> dataset = TextDataset(
    ...     data_path="data/train.json",
    ...     tokenizer=tokenizer,
    ...     max_length=512
    ... )
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        text_column: str = "text"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        
        # Load data
        self.data = self._load_data(data_path)
    
    def _load_data(self, path: str) -> List[Dict]:
        """Load data from JSON/JSONL file."""
        data = []
        
        with open(path, 'r', encoding='utf-8') as f:
            if path.endswith('.jsonl'):
                for line in f:
                    data.append(json.loads(line))
            else:
                data = json.load(f)
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item[self.text_column]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


class InstructionDataset(Dataset):
    """
    Dataset for instruction-following tasks.
    
    Format: {"instruction": "...", "input": "...", "output": "..."}
    """
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format instruction
        prompt = self._format_instruction(item)
        
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }
    
    @staticmethod
    def _format_instruction(item: Dict) -> str:
        """Format instruction into prompt."""
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        return prompt
