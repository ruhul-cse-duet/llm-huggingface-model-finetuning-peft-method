"""
Prompt Tuning PEFT Technique
=============================

Prompt Tuning adds trainable soft prompts only to the input embeddings layer,
making it the most parameter-efficient PEFT method.

Key Concepts:
- Only input layer embeddings are augmented with trainable tokens
- Most parameter-efficient: ~0.001-0.01% trainable parameters
- Scales exceptionally well with model size (best for >10B models)
- Task-specific prompt learning

Architecture:
    Input: [soft_prompt_1, ..., soft_prompt_n, x_1, x_2, ..., x_m]
    where soft_prompt_i are trainable continuous embeddings

Comparison with Prefix Tuning:
- Lighter: Only modifies input layer (vs all layers)
- Better for large models: Effectiveness improves with model size
- Simpler: No reparameterization needed

Use Cases:
- Multi-task learning (one prompt per task)
- Few-shot learning scenarios
- Very large models (10B+ parameters)
- Best for: Massive models where even 0.1% is significant

Memory: ~0.001-0.01% trainable parameters
"""

from .prompt_tuning_model import PromptTuningModel
from .prompt_tuning_config import PromptTuningConfig

__all__ = ['PromptTuningModel', 'PromptTuningConfig']
