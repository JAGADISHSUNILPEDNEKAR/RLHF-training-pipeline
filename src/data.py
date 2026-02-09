from datasets import Dataset
import json

def prepare_dataset(preferences):
    """
    Convert a list of preference dictionaries to a Hugging Face Dataset.
    
    Args:
        preferences (list): List of dicts with keys 'prompt', 'chosen', 'rejected'.
        
    Returns:
        Dataset: Hugging Face Dataset object.
    """
    data = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }

    for p in preferences:
        data["prompt"].append(p["prompt"])
        data["chosen"].append(p["chosen"])
        data["rejected"].append(p["rejected"])

    return Dataset.from_dict(data)

def load_preferences(file_path):
    """Load preferences from a JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def tokenize_fn(examples, tokenizer, max_length=256):
    """
    Tokenize examples for reward model training.
    
    Args:
        examples (dict): Batch of examples.
        tokenizer (AutoTokenizer): Tokenizer instance.
        max_length (int): Maximum sequence length.
        
    Returns:
        dict: Tokenized 'chosen' and 'rejected' inputs.
    """
    chosen = [
        p + " " + c
        for p, c in zip(examples["prompt"], examples["chosen"])
    ]

    rejected = [
        p + " " + r
        for p, r in zip(examples["prompt"], examples["rejected"])
    ]

    tok_c = tokenizer(
        chosen,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    tok_r = tokenizer(
        rejected,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    return {
        "chosen_input_ids": tok_c["input_ids"],
        "chosen_attention_mask": tok_c["attention_mask"],
        "rejected_input_ids": tok_r["input_ids"],
        "rejected_attention_mask": tok_r["attention_mask"]
    }
