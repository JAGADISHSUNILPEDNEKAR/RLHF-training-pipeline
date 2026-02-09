import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from trl import AutoModelForCausalLMWithValueHead
from peft import prepare_model_for_kbit_training
from .config import config

def load_tokenizer(model_name=config.model_name):
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model(model_name=config.model_name):
    """Load the base model for annotation/generation."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model

def load_reward_model(model_name="distilbert-base-uncased", device="cuda"):
    """Load the reward model for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1
    ).to(device)
    return model, tokenizer

def load_dpo_model(model_name=config.model_name):
    """Load the model for DPO training."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model

def load_ppo_model(model_name=config.model_name, device="cuda"):
    """Load the model with value head for PPO training."""
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device
    )
    return model
