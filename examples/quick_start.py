"""
Quick Start Example for PEFT Fine-tuning
----------------------------------------

Quick start example showing basic usage

This example demonstrates:
1. Loading a model with LoRA
2. Preparing a simple dataset
3. Training the model
4. Saving and loading fine-tuned model
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset


def main():
    print("=" * 60)
    print("PEFT Fine-tuning Quick Start")
    print("=" * 60)
    
    # Step 1: Load Model and Tokenizer
    print("\n[Step 1] Loading model and tokenizer...")
    model_name = "facebook/opt-350m"  # Small model for quick testing
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Step 2: Configure LoRA
    print("\n[Step 2] Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,  # Rank
        lora_alpha=32,  # Alpha
        target_modules=["q_proj", "v_proj"],  # Which layers
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Step 3: Prepare Dataset
    print("\n[Step 3] Loading dataset...")
    # Using a small subset for quick demo
    dataset = load_dataset("imdb", split="train[:100]")
    
    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding="max_length",
        )

    tokenized_dataset = (
        dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
    )
    
    # Step 4: Setup Training
    print("\n[Step 4] Setting up training...")
    training_args = TrainingArguments(
        output_dir="./output/quick_start",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=5,
        save_steps=50,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    # Step 5: Train
    print("\n[Step 5] Training...")
    trainer.train()
    
    # Step 6: Save Model
    print("\n[Step 6] Saving model...")
    model.save_pretrained("./output/quick_start/final_model")
    tokenizer.save_pretrained("./output/quick_start/final_model")
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print("Model saved to: ./output/quick_start/final_model")
    print("=" * 60)
    
    # Step 7: Load and Test (Bonus)
    print("\n[Step 7] Testing inference...")
    from peft import PeftModel
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(
        base_model,
        "./output/quick_start/final_model"
    )
    
    # Test generation
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    print("\n✅ All done! Check the output directory for saved models.")


if __name__ == "__main__":
    main()
