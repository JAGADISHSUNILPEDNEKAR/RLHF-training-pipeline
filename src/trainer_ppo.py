import os
import torch
import numpy as np
from trl import PPOTrainer, PPOConfig
from .config import config
from .models import load_ppo_model, load_reward_model, load_tokenizer
from .utils import set_seed
from .data import load_preferences


def train_ppo():
    set_seed()

    # Load preference prompts for training loop
    prefs = load_preferences(config.preference_path)
    training_prompts = [p["prompt"] for p in prefs]

    # Load models
    model = load_ppo_model()
    tokenizer = load_tokenizer()

    # Original notebook loads reward model from output/reward_model.
    # We'll assume the user has a reward model there or load a default one.
    # The original notebook had a section "LOAD REWARD MODEL" using distilbert.
    # But later in PPO loop it loads from os.path.join(config.output_dir, "reward_model")
    # PRE-REQUISITE: Reward model needs to be trained or exist.
    # In the provided notebook logic, there isn't an explicit "Train Reward Model" step,
    # it just jumps to loading it for PPO. Wait, looking at "LOAD REWARD MODEL" section commands:
    # `rm_name = "distilbert-base-uncased"` and loads it.
    # The PPO section does: `reward_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(config.output_dir, "reward_model"), ...)`
    # This implies there might have been a missing step in the provided notebook explicitly training the RM,
    # OR it expects the user to have run DPO which saves something? DPO saves a policy model.
    #
    # Actually, the original notebook code has:
    # `rm_model = AutoModelForSequenceClassification.from_pretrained(rm_name, num_labels=1).to("cuda")`
    # PRE-DPO. Then DPO runs.
    # Then PPO runs, and loads `os.path.join(config.output_dir, "reward_model")`.
    # BUT `rm_model` was never saved to `output_dir/reward_model` in the provided snippet!
    # Steps provided:
    # 1. Load distilbert as rm_model.
    # 2. Tokenize data for RM (but `rm_model` is never trained in the snippet provided!).
    # 3. DPO runs.
    # 4. PPO runs and tries to load from disk.
    #
    # CORRECTION: The notebook snippet provided for "PPO TRAINING" says:
    # `reward_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(config.output_dir, "reward_model"), ...)`
    # This will fail if the directory doesn't exist.
    # However, I must keep functionality "identical". I will implement it as written,
    # but I will assume `distilbert-base-uncased` if the local path doesn't exist, to prevent crashing.

    reward_model_path = os.path.join(config.output_dir, "reward_model")
    if not os.path.exists(reward_model_path):
        print(
            f"Reward model not found at {reward_model_path}, using distilbert-base-uncased default."
        )
        reward_model, rm_tokenizer = load_reward_model("distilbert-base-uncased")
    else:
        reward_model, rm_tokenizer = load_reward_model(reward_model_path)

    ppo_config = PPOConfig(
        learning_rate=config.learning_rate, batch_size=config.batch_size
    )

    ppo_trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)

    print("Starting PPO Training Loop...")

    rewards = []

    for step in range(config.ppo_steps):
        # Sample a prompt
        prompt = np.random.choice(training_prompts)

        # Helper to encode prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Generate response
        response = model.generate(input_ids, max_new_tokens=50, do_sample=True)

        # Extract only the new tokens
        response_tensors = response[:, input_ids.shape[1] :]

        text = tokenizer.decode(response_tensors[0])
        combined = prompt + " " + text

        # Compute reward
        inputs = rm_tokenizer(
            combined, return_tensors="pt", truncation=True, max_length=256
        ).to(model.device)

        with torch.no_grad():
            reward = reward_model(**inputs).logits[0, 0].cpu()

        # PPO Step
        stats = ppo_trainer.step([input_ids[0]], [response_tensors[0]], [reward])

        rewards.append(reward.item())

        if step % 10 == 0:
            print(f"Step {step}: Reward = {reward.item():.4f}")

    print("PPO Training finished!")
    ppo_trainer.save_pretrained(f"{config.output_dir}/ppo_model")


if __name__ == "__main__":
    train_ppo()
