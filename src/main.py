import os
import sys
import torch

# Add src to sys.path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.generator.generator import ModelBasedGenerator, TemplateGenerator
from src.trainer.trainer import SelfTrainingTrainer
from src.evaluator.evaluator import ModelEvaluator

class SelfTrainingOrchestrator:
    """
    Orchestrates the self-training loop: Generation -> Training -> Evaluation.
    """

    def __init__(
        self,
        base_model_name: str,
        iterations: int = 2,
        samples_per_iteration: int = 10,
        output_dir: str = "models/self_trained_model"
    ):
        self.base_model_name = base_model_name
        self.iterations = iterations
        self.samples_per_iteration = samples_per_iteration
        self.output_dir = output_dir

        # Use MPS if available
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Orchestrator initialized. Device: {self.device}")

    def run(self):
        """Executes the full self-training loop."""
        print(f"Starting self-training loop for {self.iterations} iterations.")

        current_model_name = self.base_model_name

        for i in range(self.iterations):
            print(f"\n=== Iteration {i+1}/{self.iterations} ===")

            # 1. GENERATION
            print("Step 1: Generating synthetic data...")
            # We use the model we just trained (or the base model in iteration 0) to generate data
            generator = ModelBasedGenerator(model_name=current_model_name, device=self.device)
            synthetic_samples = generator.generate(num_samples=self.samples_per_iteration)
            print(f"Generated {len(synthetic_samples)} samples.")

            # 2. TRAINING
            print("Step 2: Training on synthetic data...")
            iteration_output_dir = os.path.join(self.output_dir, f"iteration_{i+1}")
            trainer = SelfTrainingTrainer(
                model_name=current_model_name,
                output_dir=iteration_output_dir,
                num_epochs=1,
                batch_size=2,
                max_length=256
            )
            trainer.train(synthetic_samples)

            # Update current model to the newly trained one for the next iteration
            current_model_name = iteration_output_dir

            # 3. EVALUATION
            print("Step 3: Evaluating model performance...")
            evaluator = ModelEvaluator(model_path=current_model_name, device=self.device)

            # Test generation quality
            test_prompts = [
                "Tell me a joke.",
                "How do I bake a cake?",
                "The meaning of life is"
            ]
            eval_results = evaluator.evaluate_generation(test_prompts)
            print("Generation Evaluation:")
            for res in eval_results:
                print(f"  Prompt: {res['prompt']} -> Response: {res['response']}")

            # Test perplexity (on some fixed text to see improvement)
            test_texts = [
                "Artificial intelligence is transforming the world through machine learning.",
                "The cat sat on the mat and looked at the sun.",
                "Python is a versatile programming language used in many fields."
            ]
            perplexity = evaluator.calculate_perplexity(test_texts)
            print(f"Perplexity on test set: {perplexity:.4f}")

        print("\nSelf-training loop completed successfully!")
        print(f"Final model located at: {current_model_name}")

if __name__ == "__main__":
    # For a real run, use a larger model and more iterations.
    # For this foundation smoke test, we use distilgpt2.
    orchestrator = SelfTrainingOrchestrator(
        base_model_name="distilgpt2",
        iterations=1, # Set to 1 for a quick verification of the pipeline
        samples_per_iteration=4,
        output_dir="models/self_training_foundation"
    )
    orchestrator.run()
