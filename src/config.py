import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration for RLHF Pipeline."""

    # Model config
    model_name: str = "gpt2"
    use_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Training config
    batch_size: int = 4
    learning_rate: float = 1.41e-5
    rm_epochs: int = 3
    ppo_steps: int = 100
    kl_penalty: float = 0.1

    # Annotation config
    num_annotations: int = 50

    # Paths
    output_dir: str = "output"
    data_dir: str = "data"
    preference_file: str = "preferences.json"

    def __post_init__(self):
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def preference_path(self) -> str:
        return os.path.join(self.data_dir, self.preference_file)


# Global config instance
config = Config()
