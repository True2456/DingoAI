import os
import sys
from typing import List, Optional

# Add mlx_foundation/src to sys.path so we can import our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from generator.generator import MLXGenerator, MultiTeacherMLXGenerator, EnsembleAgenticTrajectoryGenerator
from trainer.trainer import MLXTrainer
from evaluator.evaluator import MLXEvaluator

class MLXSelfTrainingOrchestrator:
    """
    Orchestrates the MLX-optimized self-training loop: Generation -> Training -> Evaluation.
    """

    def __init__(
        self,
        base_model_path: str,
        iterations: int = 1,
        samples_per_iteration: int = 2,
        output_dir: str = "models/mlx_self_training",
        generator_type: str = "agentic", # options: "mlx", "multi_teacher", "agentic"
        training_iters: int = 15,
        resume_adapter_path: Optional[str] = None
    ):
        self.base_model_path = base_model_path
        self.iterations = iterations
        self.samples_per_iteration = samples_per_iteration
        self.output_dir = output_dir
        self.generator_type = generator_type
        self.training_iters = training_iters
        self.resume_adapter_path = resume_adapter_path

    def _get_generator(self, model_path: str, adapter_path: Optional[str] = None, teacher_paths: Optional[List[str]] = None):
        if self.generator_type == "multi_teacher":
            paths = teacher_paths if teacher_paths else [model_path]
            return MultiTeacherMLXGenerator(model_paths=paths)
        elif self.generator_type == "agentic":
            # For agentic ensemble training, we use the teacher paths (Qwen and Gemma 31B) to generate traces
            paths = teacher_paths if teacher_paths else [model_path]
            return EnsembleAgenticTrajectoryGenerator(model_paths=paths)
        else:
            return MLXGenerator(model_path=model_path, adapter_path=adapter_path)

    def run(self, teacher_paths: Optional[List[str]] = None):
        """Executes the full self-training loop."""
        print(f"Starting MLX self-training loop ({self.generator_type} mode) for {self.iterations} iterations.")

        current_adapter_path = self.resume_adapter_path
        if current_adapter_path:
            print(f"Resuming self-training loop starting with adapter weights from: {current_adapter_path}")

        for i in range(self.iterations):
            print(f"\n=== MLX Iteration {i+1}/{self.iterations} ===")

            # 1. GENERATION
            print(f"Step 1: Generating synthetic data using {self.generator_type} generator...")
            generator = self._get_generator(self.base_model_path, adapter_path=current_adapter_path, teacher_paths=teacher_paths)
            synthetic_samples = generator.generate(num_samples=self.samples_per_iteration)
            print(f"Generated {len(synthetic_samples)} samples.")

            # 2. TRAINING
            print("Step 2: Training on synthetic data (LoRA)...")
            iteration_output_dir = os.path.join(self.output_dir, f"iteration_{i+1}")
            
            # If resuming and it's the first step of this run, we output to iteration_1 but read from current_adapter_path.
            # If we already progressed to iteration_X, trainer reads from current_adapter_path.
            trainer = MLXTrainer(
                model_path=self.base_model_path,
                output_dir=iteration_output_dir,
                iters=self.training_iters,
                batch_size=1,
                adapter_path=current_adapter_path
            )
            trainer.train(synthetic_samples)

            # Update current adapter path to the newly trained LoRA adapter
            current_adapter_path = iteration_output_dir

            # 3. EVALUATION
            print("Step 3: Evaluating model performance...")
            evaluator = MLXEvaluator(model_path=self.base_model_path, adapter_path=current_adapter_path)

            # Test agentic syntax
            test_tasks = [
                "Write a Python script that multiplies 12 and 15 and prints the product.",
                "Create a Python script that counts the letters in 'antigravity'."
            ]
            syntax_results = evaluator.evaluate_agentic_syntax(test_tasks)
            print("Agentic Syntax Evaluation:")
            for k, v in syntax_results.items():
                print(f"  {k}: {v}")

            # Test generation quality
            test_prompts = [
                "Explain the significance of the Magna Carta.",
                "What is the capital of Japan?"
            ]
            eval_results = evaluator.evaluate_generation(test_prompts)
            print("Generation Evaluation:")
            for res in eval_results:
                print(f"  Prompt: {res['prompt']} -> Response: {res['response']}")

            # Test perplexity
            test_texts = [
                "Artificial intelligence is transforming the world through machine learning.",
                "Python is a versatile programming language used in many fields."
            ]
            perplexity = evaluator.calculate_perplexity(test_texts)
            print(f"Perplexity on test set: {perplexity:.4f}")

        print("\nMLX self-training loop completed successfully!")
        print(f"Final model/adapter located at: {current_adapter_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MLX Self-Training/Distillation Loop.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "full"],
        default="smoke",
        help="Run mode: 'smoke' for a quick validation, 'full' for complete training."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to an existing adapter folder to resume training from."
    )
    args = parser.parse_args()

    # Define paths
    base_model = "/Users/true/.lmstudio/models/gemma-4-26b-a4b-it-oQ8"
    teacher_paths = [
        "/Users/true/.lmstudio/models/alvarolizama/Qwen3.6-35B-A3B-oQ8-mtp",
        "/Users/true/.lmstudio/models/gemma-4-31b-it-oQ8"
    ]

    if args.mode == "smoke":
        print("Configuring loop for SMOKE TEST...")
        orchestrator = MLXSelfTrainingOrchestrator(
            base_model_path=base_model,
            iterations=1,
            samples_per_iteration=1,
            generator_type="agentic",
            training_iters=15,
            resume_adapter_path=args.resume
        )
    else:
        print("Configuring loop for FULL DISTILLATION RUN...")
        orchestrator = MLXSelfTrainingOrchestrator(
            base_model_path=base_model,
            iterations=3,
            samples_per_iteration=20,
            generator_type="agentic",
            training_iters=500,
            resume_adapter_path=args.resume
        )

    orchestrator.run(teacher_paths=teacher_paths)
