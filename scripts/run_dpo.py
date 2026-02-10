import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trainer_dpo import train_dpo

if __name__ == "__main__":
    train_dpo()
