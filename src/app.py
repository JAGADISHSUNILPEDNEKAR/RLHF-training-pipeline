import gradio as gr
import json
import os
import torch
from threading import Thread
import subprocess
from .config import config
from .models import load_base_model, load_tokenizer

# Global state for the model (lazy loading recommended for faster app startup)
model = None
tokenizer = None


def get_model():
    global model, tokenizer
    if model is None:
        model = load_base_model()
        tokenizer = load_tokenizer()
    return model, tokenizer


# --- Chat Interface ---
def generate_chat_response(message, history):
    model, tokenizer = get_model()

    # Format history (simplified for GPT-2)
    prompt = ""
    for user_msg, bot_msg in history:
        prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"
    prompt += f"User: {message}\nAssistant:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the new part
    try:
        response = response.split("Assistant:")[-1].strip()
    except:
        pass

    return response


# --- Annotation Interface ---
# Sample prompts
PROMPTS = [
    "Explain quantum computing to a 5-year-old.",
    "Write a poem about a robot who loves flowers.",
    "What are the benefits of exercise?",
    "How do I make a cake?",
    "Tell me a joke.",
]
current_prompt_index = 0
annotation_data = []


def get_next_prompt():
    global current_prompt_index
    if current_prompt_index >= len(PROMPTS):
        return "Annotation Complete!", "", ""

    prompt = PROMPTS[current_prompt_index]

    model, tokenizer = get_model()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    responses = []
    for _ in range(2):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
        resp = resp[len(prompt) :].strip()  # Remove prompt
        responses.append(resp)

    return prompt, responses[0], responses[1]


def save_preference(choice, prompt, resp_a, resp_b):
    global current_prompt_index, annotation_data

    if choice == "A is Better":
        chosen = resp_a
        rejected = resp_b
    elif choice == "B is Better":
        chosen = resp_b
        rejected = resp_a
    else:  # Tie
        current_prompt_index += 1
        return get_next_prompt()

    annotation_data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    # Save to file
    os.makedirs(config.data_dir, exist_ok=True)
    with open(config.preference_path, "w") as f:
        json.dump(annotation_data, f, indent=2)

    current_prompt_index += 1
    return get_next_prompt()


# --- Training Interface ---
def run_training_script(script_name):
    # This is a simple way to run the script.
    # In a real deployed app, you'd want a more robust task queue (e.g. Celery)
    try:
        subprocess.Popen(["python", f"scripts/{script_name}"])
        return f"Started {script_name}. Check terminal for logs."
    except Exception as e:
        return f"Error starting training: {str(e)}"


# --- Gradio App Layout ---
theme = gr.themes.Soft()

with gr.Blocks(theme=theme, title="RLHF Training Pipeline") as app:
    gr.Markdown("# 🚀 RLHF Training Pipeline")

    with gr.Tab("💬 Chat"):
        gr.ChatInterface(generate_chat_response)

    with gr.Tab("✍️ Annotation"):
        gr.Markdown("### Compare two model responses and vote for the best one.")

        with gr.Row():
            prompt_box = gr.Textbox(label="Prompt", interactive=False)

        with gr.Row():
            resp_a_box = gr.Textbox(label="Response A", interactive=False, lines=5)
            resp_b_box = gr.Textbox(label="Response B", interactive=False, lines=5)

        with gr.Row():
            btn_a = gr.Button("A is Better", variant="primary")
            btn_tie = gr.Button("Tie / Skip", variant="secondary")
            btn_b = gr.Button("B is Better", variant="primary")

        # Initial load (hacky way to load first prompt)
        # We use a hidden button to trigger the first load
        load_btn = gr.Button("Start Annotation", visible=True)

        def start_annotation():
            p, a, b = get_next_prompt()
            return {
                prompt_box: p,
                resp_a_box: a,
                resp_b_box: b,
                load_btn: gr.update(visible=False),
            }

        load_btn.click(
            start_annotation, outputs=[prompt_box, resp_a_box, resp_b_box, load_btn]
        )

        btn_a.click(
            save_preference,
            inputs=[gr.State("A is Better"), prompt_box, resp_a_box, resp_b_box],
            outputs=[prompt_box, resp_a_box, resp_b_box],
        )
        btn_b.click(
            save_preference,
            inputs=[gr.State("B is Better"), prompt_box, resp_a_box, resp_b_box],
            outputs=[prompt_box, resp_a_box, resp_b_box],
        )
        btn_tie.click(
            save_preference,
            inputs=[gr.State("Tie"), prompt_box, resp_a_box, resp_b_box],
            outputs=[prompt_box, resp_a_box, resp_b_box],
        )

    with gr.Tab("⚙️ Training"):
        gr.Markdown("### Train your model")
        gr.Markdown(
            "Once you have collected enough preferences, you can start the training process."
        )

        with gr.Row():
            dpo_btn = gr.Button("Start DPO Training", variant="primary")
            ppo_btn = gr.Button("Start PPO Training", variant="primary")

        status_box = gr.Textbox(label="Status", interactive=False)

        dpo_btn.click(lambda: run_training_script("run_dpo.py"), outputs=status_box)
        ppo_btn.click(lambda: run_training_script("run_ppo.py"), outputs=status_box)


# Expose the app object for the root script
app_interface = app

if __name__ == "__main__":
    app.launch()
