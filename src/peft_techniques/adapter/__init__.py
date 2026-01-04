"""
Adapter Layers PEFT Technique
==============================

Adapters insert small bottleneck layers within transformer blocks,
allowing task-specific adaptation while keeping most parameters frozen.

Key Concepts:
- Small neural networks inserted into each transformer layer
- Typically placed after attention and feed-forward sub-layers
- Uses bottleneck architecture (down-project → activation → up-project)
- Residual connections preserve original model behavior

Architecture:
    h = Adapter(x) + x
    Adapter(x) = W_up · σ(W_down · x)
    
    where:
    - W_down: Projects to bottleneck dimension (e.g., 64)
    - σ: Activation function (usually ReLU or GELU)
    - W_up: Projects back to original dimension

Advantages:
- Modular: Can stack multiple adapters for multi-task learning
- Sequential training: Add new adapters without forgetting
- Inference efficient: Can dynamically switch adapters

Use Cases:
- Multi-task learning
- Domain adaptation  
- Continual learning
- Best for: Learning multiple related tasks sequentially

Memory: ~0.5-5% trainable parameters
"""

from .adapter_model import AdapterModel
from .adapter_config import AdapterConfig

__all__ = ['AdapterModel', 'AdapterConfig']
