import torch
from transformers import TrainingArguments
from trl import DPOTrainer
from .config import config
from .data import prepare_dataset, load_preferences
from .models import load_dpo_model, load_tokenizer
from .utils import set_seed

def train_dpo():
    set_seed()
    
    # Load data
    prefs = load_preferences(config.preference_path)
    dataset = prepare_dataset(prefs)
    split = dataset.train_test_split(test_size=0.2)
    train_ds = split["train"]
    eval_ds = split["test"]
    
    # Load model & tokenizer
    model = load_dpo_model()
    tokenizer = load_tokenizer()
    
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.rm_epochs,
        per_device_train_batch_size=2, # As per original notebook
        learning_rate=5e-5,
        fp16=True,
        logging_steps=5,
        save_steps=50,
        report_to="none"
    )
    
    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        beta=0.1
    )
    
    print("Starting DPO Training...")
    dpo_trainer.train()
    print("DPO Training finished!")
    
    # Save model
    dpo_trainer.save_model(f"{config.output_dir}/dpo_model")

if __name__ == "__main__":
    train_dpo()
