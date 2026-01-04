"""
IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
=====================================================================

IA³ is an extremely lightweight PEFT method that rescales activations with
learned vectors, achieving strong performance with minimal parameters.

Key Concepts:
- Learn scalar vectors to rescale activations (element-wise multiplication)
- Three sets of vectors: l_k, l_v (attention), l_ff (feed-forward)
- No additional layers or matrix multiplications
- Zero inference latency overhead

Architecture:
    For attention: K' = l_k ⊙ K,  V' = l_v ⊙ V
    For FFN: h' = l_ff ⊙ h
    
    where ⊙ is element-wise multiplication
    l_k, l_v, l_ff are learned vectors

Advantages:
- Most parameter-efficient after Prompt Tuning (~0.01%)
- Zero inference latency (unlike adapters)
- Simple implementation
- Strong empirical performance

Use Cases:
- Production systems (no latency penalty)
- Very large scale deployments
- When inference speed is critical
- Best for: Latency-sensitive applications

Memory: ~0.01% trainable parameters
Inference Overhead: ZERO (just element-wise multiplication)
"""

from .ia3_model import IA3Model
from .ia3_config import IA3Config

__all__ = ['IA3Model', 'IA3Config']
