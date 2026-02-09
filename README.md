# RLHF Training Pipeline

This repository contains a modular implementation of a Reinforcement Learning from Human Feedback (RLHF) pipeline, migrated from a Google Colab notebook. It supports Annotation, DPO (Direct Preference Optimization), and PPO (Proximal Policy Optimization) training.

## Google Colab Link 
https://colab.research.google.com/drive/1FRDL6FjOqkULM8Rzs3_Niqkws_zeOvCt

## Directory Structure

```
rlhf-pipeline/
├── config/             # Configuration files
├── data/               # Data storage
├── notebooks/          # Jupyter notebooks for UI
├── output/             # Model outputs
├── scripts/            # CLI Execution scripts
├── src/                # Source code
│   ├── annotation.py   # Annotation UI logic
│   ├── config.py       # Configuration settings
│   ├── data.py         # Data processing
│   ├── models.py       # Model loading
│   ├── trainer_dpo.py  # DPO training
│   ├── trainer_ppo.py  # PPO training
│   └── utils.py        # Utilities
├── requirements.txt    # Dependencies
└── README.md           # Documentation
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd rlhf-pipeline
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Annotation
You can run the annotation tool in two ways:

**Option A: Jupyter Notebook (Recommended)**
Open `notebooks/annotation.ipynb` in Jupyter Lab or Notebook to use the interactive internal UI.

**Option B: CLI**
Run the command-line interface:
```bash
python scripts/run_annotation.py
```
This will start an interactive session where you can prefer Response A or B.

### 2. DPO Training
Once you have collected preferences (saved to `data/preferences.json`), run DPO training:

```bash
python scripts/run_dpo.py
```
This will train a model using the collected preferences and save it to `output/dpo_model`.

### 3. PPO Training
To run PPO training:

```bash
python scripts/run_ppo.py
```
This will load a reward model and fine-tune using PPO.

## Configuration
Modify `src/config.py` to change parameters like:
- `model_name` (default: "gpt2")
- `batch_size`
- `learning_rate`
- `output_dir`
