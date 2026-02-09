from src.annotation import AnnotationUI
import sys

# Since AnnotationUI uses ipywidgets, it won't work in a standard terminal.
# We'll create a simple CLI subclass or alternative here for terminal usage.

class CLIAnnotation(AnnotationUI):
    def _setup_ui(self):
        print("Starting CLI Annotation...")
        pass

    def display(self):
        print("\n=== RLHF Annotation CLI ===\n")
        
        while self.current_index < len(self.sample_prompts):
            prompt = self.sample_prompts[self.current_index]
            print(f"\nPrompt [{self.current_index + 1}/{len(self.sample_prompts)}]: {prompt}")
            print("Generating responses...")
            
            responses = self._generate_responses(prompt)
            
            print(f"\n[A]: {responses[0]}")
            print(f"\n[B]: {responses[1]}")
            
            while True:
                choice = input("\nWhich is better? (A/B/T for Tie): ").strip().upper()
                if choice in ['A', 'B', 'T']:
                    self._on_click(choice)
                    break
                print("Invalid choice. Please enter A, B, or T.")
                
        print("\nAnnotation Complete! Preferences saved.")

if __name__ == "__main__":
    app = CLIAnnotation()
    app.display()
