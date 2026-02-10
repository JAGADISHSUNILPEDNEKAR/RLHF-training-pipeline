import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import os
import torch
from .config import config
from .models import load_base_model, load_tokenizer


class AnnotationUI:
    def __init__(self):
        self.model = load_base_model()
        self.tokenizer = load_tokenizer()
        self.preferences = []
        self.current_index = 0

        # Sample prompts (could be loaded from a file/config)
        self.sample_prompts = [
            "Explain what artificial intelligence is in simple terms.",
            "Write a short story about a robot learning to paint.",
            "How does the internet work?",
            "What is happiness?",
            "Explain gravity in simple words.",
            "Describe your dream job.",
            "What makes life meaningful?",
            "Explain the solar system.",
        ]

        self._setup_ui()

    def _generate_responses(self, prompt, num_responses=2, max_length=100):
        responses = []
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(
            self.model.device
        )

        for _ in range(num_responses):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.9,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt) :].strip()
            responses.append(response)

        return responses

    def _setup_ui(self):
        self.output_area = widgets.Output()
        self.prompt_display = widgets.HTML()

        self.response_a = widgets.Textarea(
            disabled=True, layout=widgets.Layout(width="100%", height="150px")
        )
        self.response_b = widgets.Textarea(
            disabled=True, layout=widgets.Layout(width="100%", height="150px")
        )

        self.progress_label = widgets.HTML()

        self.btn_a = widgets.Button(description="A is Better", button_style="success")
        self.btn_b = widgets.Button(description="B is Better", button_style="success")
        self.btn_tie = widgets.Button(description="Tie/Skip", button_style="warning")

        self.btn_a.on_click(lambda b: self._on_click("A"))
        self.btn_b.on_click(lambda b: self._on_click("B"))
        self.btn_tie.on_click(lambda b: self._on_click("T"))

        button_box = widgets.HBox([self.btn_a, self.btn_tie, self.btn_b])

        self.ui_container = widgets.VBox(
            [
                widgets.HTML("<h2>RLHF Annotation</h2>"),
                self.progress_label,
                self.prompt_display,
                widgets.HTML("Response A"),
                self.response_a,
                widgets.HTML("Response B"),
                self.response_b,
                button_box,
            ]
        )

    def _save_preferences(self):
        with open(config.preference_path, "w") as f:
            json.dump(self.preferences, f, indent=2)
        print("Preferences saved!")

    def _update_display(self):
        if self.current_index >= len(self.sample_prompts):
            self.prompt_display.value = "<h3>Annotation Complete</h3>"
            self.btn_a.disabled = True
            self.btn_b.disabled = True
            self.btn_tie.disabled = True
            self._save_preferences()
            return

        prompt = self.sample_prompts[self.current_index]
        # In a real app, generate asynchronously or cache. Here we generate on the fly which blocks UI.
        # But this matches the Colab notebook behavior.
        responses = self._generate_responses(prompt)

        self.prompt_display.value = f"<h3>{prompt}</h3>"
        self.response_a.value = responses[0]
        self.response_b.value = responses[1]
        self.progress_label.value = f"Progress: {len(self.preferences)}"

    def _on_click(self, choice):
        prompt = self.sample_prompts[self.current_index]

        if choice == "A":
            self.preferences.append(
                {
                    "prompt": prompt,
                    "chosen": self.response_a.value,
                    "rejected": self.response_b.value,
                }
            )
        elif choice == "B":
            self.preferences.append(
                {
                    "prompt": prompt,
                    "chosen": self.response_b.value,
                    "rejected": self.response_a.value,
                }
            )

        self.current_index += 1
        self._update_display()

    def display(self):
        self._update_display()
        display(self.ui_container)
