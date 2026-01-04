# 🚀 LLM Fine-Tuning with PEFT Techniques
### Parameter-Efficient Fine-Tuning for Large Language Models

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents
- [বাংলা বর্ণনা (Bengali Description)](#-বাংলা-বর্ণনা)
- [English Description](#-english-description)
- [PEFT Techniques Overview](#-peft-techniques-overview)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Examples](#-usage-examples)
- [Contributing](#-contributing)

---

## বাংলা বর্ণনা

### প্রজেক্ট সম্পর্কে
এই প্রজেক্টটি হলো একটি সম্পূর্ণ **Modularized এবং Production-Ready** সিস্টেম যা Large Language Models (LLM) গুলোকে **Parameter-Efficient Fine-Tuning (PEFT)** টেকনিক ব্যবহার করে Fine-tune করার জন্য তৈরি।

### কেন PEFT ব্যবহার করবেন?
- ⚡ **কম Memory ব্যবহার**: সম্পূর্ণ model এর বদলে শুধু কিছু parameters update হয়
- 💰 **খরচ সাশ্রয়ী**: কম computational resources প্রয়োজন
- 🚀 **দ্রুত Training**: traditional fine-tuning এর চেয়ে অনেক দ্রুত
- 🎯 **Better Performance**: specific tasks এ excellent results
- 💾 **ছোট Model Size**: শুধু adapter weights save করতে হয়

---

## 🌐 English Description

### About This Project
This is a comprehensive, **production-ready framework** for fine-tuning Large Language Models using various Parameter-Efficient Fine-Tuning (PEFT) techniques.

### Key Features
- 📦 **Modular Architecture**: Easy to extend and customize
- 🔧 **7 PEFT Techniques**: LoRA, QLoRA, Prefix Tuning, and more
- 📊 **Built-in Monitoring**: Track training metrics and performance
- 🎨 **Flexible Configuration**: YAML-based config system
- 🧪 **Testing Suite**: Comprehensive unit tests
- 📚 **Rich Documentation**: Detailed guides and examples

---

## 🎓 PEFT Techniques Overview

### 1. LoRA (Low-Rank Adaptation) 🔥
**বাংলা বর্ণনা:**
- সবচেয়ে জনপ্রিয় PEFT technique
- Model এর weight matrices তে low-rank decomposition যোগ করে
- মাত্র 0.1-1% parameters train করতে হয়
- খুব কম memory (4-8GB) তে বড় models train করা যায়

**English Description:**
- Most popular PEFT technique
- Adds trainable low-rank decomposition matrices to model weights
- Trains only 0.1-1% of parameters
- Can fine-tune large models with 4-8GB VRAM

**Mathematical Formula:**
```
W = W₀ + BA
where B ∈ R^(d×r), A ∈ R^(r×k), r << min(d,k)
```

**Use Cases:**
- Text classification, QA, summarization
- Best for: General-purpose fine-tuning

---

### 2. QLoRA (Quantized LoRA) 💎
**বাংলা বর্ণনা:**
- LoRA এর উন্নত version
- Base model কে 4-bit quantization করে memory আরও কমায়
- একটি single 24GB GPU তে 65B parameters model train করা যায়
- Quality প্রায় same থাকে কিন্তু memory 75% কম লাগে

**English Description:**
- Advanced version of LoRA
- Quantizes base model to 4-bit to reduce memory further
- Can train 65B parameter models on a single 24GB GPU
- Maintains quality while using 75% less memory

**Key Innovation:**
- 4-bit NormalFloat (NF4) quantization
- Double quantization
- Paged optimizers for memory management

**Use Cases:**
- Training very large models (13B-65B+)
- Limited GPU memory scenarios
- Best for: Resource-constrained environments

---

### 3. Prefix Tuning 🎯
**বাংলা বর্ণনা:**
- Model এর প্রতিটি layer এ trainable "prefix" vectors যোগ করে
- Actual model weights frozen থাকে
- Natural language prefix এর মতো কাজ করে কিন্তু continuous space এ
- প্রায় 0.01-1% parameters trainable

**English Description:**
- Adds trainable prefix vectors to each transformer layer
- Original model weights remain frozen
- Acts like natural language prefixes but in continuous space
- Only 0.01-1% parameters are trainable

**Architecture:**
```
h₀ = [P_θ; x]
where P_θ = trainable prefix parameters
```

**Use Cases:**
- Table-to-text generation
- Summarization
- Best for: Generation tasks

---

### 4. Prompt Tuning 📝
**বাংলা বর্ণনা:**
- শুধুমাত্র input embeddings এ soft prompts যোগ করে
- সবচেয়ে parameter-efficient method (মাত্র 0.001-0.01%)
- বড় models (>10B) এ খুব ভালো কাজ করে
- প্রতিটি task এর জন্য আলাদা prompt learn করে

**English Description:**
- Adds soft prompts only to input embeddings
- Most parameter-efficient (only 0.001-0.01% params)
- Works exceptionally well with large models (>10B)
- Learns task-specific prompts

**Comparison with Prefix Tuning:**
- Lighter weight (only input layer)
- Better for very large models
- Task-specific optimization

**Use Cases:**
- Multi-task learning
- Few-shot learning scenarios
- Best for: Models with >10B parameters

---

### 5. Adapter Layers 🔌
**বাংলা বর্ণনা:**
- প্রতিটি transformer layer এ small bottleneck layers insert করে
- Feed-forward network এর মধ্যে এবং পরে যোগ হয়
- Training এবং inference দুটোতেই efficient
- প্রায় 0.5-5% parameters add হয়

**English Description:**
- Inserts small bottleneck layers into each transformer layer
- Added within and after feed-forward networks
- Efficient for both training and inference
- Adds approximately 0.5-5% parameters

**Architecture:**
```
h = W_up(σ(W_down(x))) + x
where W_down: bottleneck dimension
```

**Use Cases:**
- Multi-task learning
- Domain adaptation
- Best for: Sequential task learning

---

### 6. IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations) ⚡
**বাংলা বর্ণনা:**
- Adapter এর চেয়েও lightweight
- Learned vectors দিয়ে activations কে rescale করে
- কোনো extra latency নেই inference এ
- মাত্র 0.01% parameters trainable

**English Description:**
- Even lighter than adapters
- Rescales activations with learned vectors
- No additional inference latency
- Only 0.01% parameters are trainable

**Key Innovation:**
- Element-wise rescaling (no matrix multiplication)
- Three sets of vectors: keys, values, and feed-forward
- Zero additional inference cost

**Use Cases:**
- Production environments (no latency increase)
- Very large scale deployments
- Best for: Latency-sensitive applications

---

### 7. P-Tuning (v2) 🎨
**বাংলা বর্ণনা:**
- Prefix Tuning এর উন্নত version
- শুধু input layer নয়, সব layers এ prompts যোগ করে
- Small থেকে large সব size models এ কাজ করে
- Natural Language Understanding tasks এ excellent

**English Description:**
- Improved version of Prefix Tuning
- Adds prompts to all layers, not just input
- Works across all model sizes
- Excellent for NLU tasks

**Architecture:**
```
[P₁, P₂, ..., Pₙ, x₁, x₂, ..., xₘ]
where Pᵢ = continuous prompt embeddings
```

**Use Cases:**
- Question Answering
- Named Entity Recognition (NER)
- Best for: Understanding tasks with smaller models

---

## 📊 PEFT Techniques Comparison

| Technique | Trainable % | Memory | Speed | Inference Latency | Best For |
|-----------|-------------|--------|-------|-------------------|----------|
| **LoRA** | 0.1-1% | Low | Fast | None | General purpose |
| **QLoRA** | 0.1-1% | Very Low | Fast | None | Large models, limited GPU |
| **Prefix Tuning** | 0.01-1% | Low | Medium | Small | Generation tasks |
| **Prompt Tuning** | 0.001-0.01% | Very Low | Fast | None | Large models (>10B) |
| **Adapter** | 0.5-5% | Medium | Medium | Small | Multi-task learning |
| **IA³** | 0.01% | Very Low | Very Fast | None | Production systems |
| **P-Tuning v2** | 0.01-0.1% | Low | Fast | Small | NLU tasks |

---

## 🏗️ Project Structure

```
LLM-PEFT-FineTuning/
│
├── src/
│   ├── peft_techniques/
│   │   ├── lora/
│   │   │   ├── __init__.py
│   │   │   ├── lora_model.py
│   │   │   └── lora_config.py
│   │   ├── qlora/
│   │   ├── prefix_tuning/
│   │   ├── prompt_tuning/
│   │   ├── adapter/
│   │   ├── ia3/
│   │   └── p_tuning/
│   ├── trainers/
│   │   ├── base_trainer.py
│   │   └── peft_trainer.py
│   ├── data_loaders/
│   │   ├── dataset.py
│   │   └── tokenizer.py
│   └── utils/
│       ├── logging.py
│       ├── metrics.py
│       └── helpers.py
│
├── configs/
│   ├── lora_config.yaml
│   ├── qlora_config.yaml
│   └── training_config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── checkpoints/
│   └── final/
│
├── notebooks/
│   ├── 01_LoRA_Tutorial.ipynb
│   ├── 02_QLoRA_Tutorial.ipynb
│   └── 03_Comparison.ipynb
│
├── docs/
│   ├── PEFT_Guide.md
│   └── API_Reference.md
│
├── tests/
│   └── test_peft.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
│
├── examples/
│   └── quick_start.py
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM (16GB+ recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/ruhul-cse-duet/llm-huggingface-model-finetuning-peft-method.git
cd llm-huggingface-model-finetuning-peft-method
```

### Step 2: Create Virtual Environment
```bash
# Using conda
conda create -n peft python=3.11
conda activate peft

# Or using venv
python -m venv peft_env
source peft_env/bin/activate  # Linux/Mac
# peft_env\Scripts\activate  # Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Example 1: LoRA Fine-tuning
```python
from src.peft_techniques.lora import LoRAModel, LoRAConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
model_name = "meta-llama/Llama-2-7b-hf"
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configure LoRA
lora_config = LoRAConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05
)

# Apply LoRA
model = LoRAModel(base_model, lora_config)

# Train
from src.trainers import PEFTTrainer
trainer = PEFTTrainer(model, tokenizer, train_dataset, eval_dataset)
trainer.train()
```

### Example 2: QLoRA Fine-tuning
```python
from src.peft_techniques.qlora import QLoRAModel, QLoRAConfig

# Configure QLoRA with 4-bit quantization
qlora_config = QLoRAConfig(
    r=16,
    lora_alpha=64,
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

model = QLoRAModel.from_pretrained(
    "meta-llama/Llama-2-13b-hf",
    qlora_config=qlora_config
)

# Train
trainer = PEFTTrainer(model, tokenizer, train_dataset)
trainer.train()
```

### Example 3: Using Configuration Files
```bash
# Train with YAML config
python scripts/train.py --config configs/lora_config.yaml

# Evaluate
python scripts/evaluate.py --model models/checkpoints/lora_model

# Inference
python scripts/inference.py --model models/final/lora_model --prompt "Your text here"
```

---

## 📚 Usage Examples

### Training with Custom Dataset
```python
from src.data_loaders import CustomDataset
from torch.utils.data import DataLoader

# Prepare your dataset
dataset = CustomDataset(
    data_path="data/raw/your_data.json",
    tokenizer=tokenizer,
    max_length=512
)

train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train
trainer = PEFTTrainer(
    model=model,
    train_dataloader=train_loader,
    epochs=3,
    learning_rate=2e-4
)
trainer.train()
```

---

## 📖 Documentation

- [Detailed PEFT Guide](docs/PEFT_Guide.md) - In-depth explanation of each technique
- [API Reference](docs/API_Reference.md) - Complete API documentation
- [Tutorials](notebooks/) - Jupyter notebooks with examples

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

---

## 📧 Contact
Ruhul Amin[https://www.linkedin.com/in/ruhul-duet-cse/]

For questions or support, please open an issue or  
contact: ruhul.cse.duet@gmail.com

---

**Happy Fine-Tuning! 🎉**
