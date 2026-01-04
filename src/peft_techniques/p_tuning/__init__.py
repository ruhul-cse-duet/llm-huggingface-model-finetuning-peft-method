"""
P-Tuning v2 PEFT Technique
===========================

P-Tuning v2 is an improved version of prefix tuning that adds prompts
to every layer and works well across all model scales.

Key Concepts:
- Extension of prefix tuning with prompts at all layers
- Unlike Prefix Tuning, optimized for understanding tasks
- Works well with both small and large models
- No reparameterization needed (unlike original P-Tuning)

Architecture:
    For each layer l:
    Input: [P_0^l, P_1^l, ..., P_n^l, x_1, x_2, ..., x_m]
    
    where P_i^l are continuous prompt embeddings per layer

Comparison with Other Methods:
- vs Prefix Tuning: Simpler (no MLP projection), works on smaller models
- vs Prompt Tuning: Deeper (all layers vs just input), better for NLU
- vs LoRA: Similar parameter efficiency, different mechanism

Use Cases:
- Question Answering
- Named Entity Recognition (NER)
- Natural Language Understanding tasks
- Smaller models (< 10B) where Prompt Tuning struggles
- Best for: NLU tasks with smaller-to-medium models

Memory: ~0.01-0.1% trainable parameters
"""

from .p_tuning_model import PTuningModel
from .p_tuning_config import PTuningConfig

__all__ = ['PTuningModel', 'PTuningConfig']
