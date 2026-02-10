import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trainer_ppo import train_ppo

if __name__ == "__main__":
    train_ppo()
