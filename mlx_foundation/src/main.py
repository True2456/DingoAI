import os
import sys
import json
from typing import List, Optional

# Add mlx_foundation/src to sys.path so we can import our modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from generator.generator import MLXGenerator, MultiTeacherMLXGenerator, EnsembleAgenticTrajectoryGenerator
from trainer.trainer import MLXTrainer
from evaluator.evaluator import MLXEvaluator


def run_generate_only(teacher_paths: List[str], num_samples: int, output_path: str):
    """
    Standalone generation mode — runs on the second machine (e.g. M3 Max 64GB).
    Loads only the teacher models, generates agentic coding trajectories, and saves
    them to a JSONL file. No student model is loaded. No training happens.

    The output file can then be copied to the training machine and fed in via
    --mode train-only --data <path>.
    """
    print(f"=== GENERATE-ONLY MODE ===")
    print(f"Generating {num_samples} agentic trajectories using teachers: {teacher_paths}")
    print(f"Output will be saved to: {output_path}")

    generator = EnsembleAgenticTrajectoryGenerator(model_paths=teacher_paths)
    samples = generator.generate(num_samples=num_samples)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")

    print(f"\nDone. {len(samples)} trajectories saved to: {output_path}")
    print("Transfer this file to your training machine and run:")
    print(f"  python src/main.py --mode train-only --data {output_path}")


def run_train_only(
    base_model_path: str,
    data_path: str,
    output_dir: str,
    training_iters: int,
    resume_adapter_path: Optional[str],
):
    """
    Standalone training mode — reads a pre-generated JSONL file and runs LoRA
    fine-tuning + evaluation. No teacher models are loaded.

    Use this on the primary machine after receiving a batch file from the
    generate-only machine.
    """
    print(f"=== TRAIN-ONLY MODE ===")
    print(f"Loading pre-generated data from: {data_path}")

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        sys.exit(1)

    with open(data_path, "r") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} samples from {data_path}.")

    # Safety check: warn if ratio is dangerous
    ratio = training_iters / max(len(samples), 1)
    if ratio > 3.0:
        print(f"WARNING: iters/samples ratio is {ratio:.1f} (>{3.0}). Risk of memorization.")
        print(f"Consider reducing --train-iters or generating more samples.")

    os.makedirs(output_dir, exist_ok=True)

    trainer = MLXTrainer(
        model_path=base_model_path,
        output_dir=output_dir,
        iters=training_iters,
        batch_size=1,
        adapter_path=resume_adapter_path
    )
    trainer.train(samples)

    print("\nEvaluating trained model...")
    evaluator = MLXEvaluator(model_path=base_model_path, adapter_path=output_dir)

    test_tasks = [
        "Write a Python script that multiplies 12 and 15 and prints the product.",
        "Create a Python script that counts the letters in 'antigravity'.",
    ]
    syntax_results = evaluator.evaluate_agentic_syntax(test_tasks)
    print("Agentic Syntax Evaluation:")
    for k, v in syntax_results.items():
        print(f"  {k}: {v}")

    test_prompts = [
        "Explain the significance of the Magna Carta.",
        "What is the capital of Japan?",
    ]
    eval_results = evaluator.evaluate_generation(test_prompts)
    print("Generation Evaluation:")
    for res in eval_results:
        print(f"  Prompt: {res['prompt']} -> Response: {res['response']}")

    perplexity = evaluator.calculate_perplexity([
        "Artificial intelligence is transforming the world through machine learning.",
        "Python is a versatile programming language used in many fields.",
    ])
    print(f"Perplexity on test set: {perplexity:.4f}")

    if perplexity > MLXSelfTrainingOrchestrator.PERPLEXITY_COLLAPSE_THRESHOLD:
        print(f"\n[COLLAPSE GATE] Perplexity {perplexity:.1f} exceeds threshold. Model may have collapsed.")
        print(f"Check your iters/samples ratio (current: {ratio:.1f}, target: <3.0).")
    else:
        print(f"\nTraining complete. Adapter saved to: {output_dir}")

class MLXSelfTrainingOrchestrator:
    """
    Orchestrates the MLX-optimized self-training loop: Generation -> Training -> Evaluation.
    """

    # Perplexity above this threshold means the model has collapsed — stop immediately.
    PERPLEXITY_COLLAPSE_THRESHOLD = 5000.0

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

            # === COLLAPSE GATE ===
            # A perplexity above the threshold means the model has catastrophically collapsed.
            # Proceeding to the next iteration would waste compute and corrupt the adapter chain.
            if perplexity > self.PERPLEXITY_COLLAPSE_THRESHOLD:
                print(f"\n[COLLAPSE GATE TRIGGERED] Perplexity {perplexity:.1f} > {self.PERPLEXITY_COLLAPSE_THRESHOLD}.")
                print(f"Model has collapsed at iteration {i+1}. Stopping training loop to prevent cascading damage.")
                print(f"Diagnosis: Most likely cause is over-training on too few samples (iters/samples ratio too high).")
                print(f"Fix: Increase samples_per_iteration OR reduce training_iters. Current ratio: {self.training_iters}/{self.samples_per_iteration} = {self.training_iters/max(self.samples_per_iteration,1):.1f} (target: <3.0)")
                break

        print("\nMLX self-training loop completed successfully!")
        print(f"Final model/adapter located at: {current_adapter_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the MLX Self-Training/Distillation Loop.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["smoke", "full", "generate-only", "train-only"],
        default="smoke",
        help=(
            "Run mode:\n"
            "  smoke         — quick 1-iteration validation (default)\n"
            "  full          — full 3-iteration distillation run\n"
            "  generate-only — generate trajectories and save to JSONL (no student model)\n"
            "  train-only    — load a pre-generated JSONL and run LoRA training (no teacher models)"
        )
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to an existing adapter folder to resume training from (smoke/full/train-only)."
    )
    # generate-only flags
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated_trajectories.jsonl",
        help="[generate-only] Path to write the generated trajectories JSONL file."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="[generate-only] Number of agentic trajectories to generate (default: 100)."
    )
    # train-only flags
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="[train-only] Path to a pre-generated JSONL file to train on."
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=200,
        help="[train-only] Number of LoRA training iterations (default: 200)."
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="models/mlx_self_training/train_only",
        help="[train-only] Directory to save the trained adapter (default: models/mlx_self_training/train_only)."
    )
    args = parser.parse_args()

    # Define paths
    base_model = "/Users/true/.lmstudio/models/gemma-4-26b-a4b-it-oQ8"
    teacher_paths = [
        "/Users/true/.lmstudio/models/lmstudio-community/Qwen3-Coder-Next-MLX-8bit",
        "/Users/true/.lmstudio/models/gemma-4-31b-it-oQ8"
    ]

    if args.mode == "generate-only":
        run_generate_only(
            teacher_paths=teacher_paths,
            num_samples=args.samples,
            output_path=args.output,
        )

    elif args.mode == "train-only":
        if not args.data:
            print("ERROR: --mode train-only requires --data <path/to/trajectories.jsonl>")
            sys.exit(1)
        run_train_only(
            base_model_path=base_model,
            data_path=args.data,
            output_dir=args.train_output,
            training_iters=args.train_iters,
            resume_adapter_path=args.resume,
        )

    elif args.mode == "smoke":
        print("Configuring loop for SMOKE TEST...")
        orchestrator = MLXSelfTrainingOrchestrator(
            base_model_path=base_model,
            iterations=1,
            samples_per_iteration=1,
            generator_type="agentic",
            training_iters=15,
            resume_adapter_path=args.resume
        )
        orchestrator.run(teacher_paths=teacher_paths)

    else:  # full
        print("Configuring loop for FULL DISTILLATION RUN...")
        # Rule of thumb: training_iters should be ~2x samples_per_iteration.
        # 100 samples * 2 = 200 iters keeps the model in the generalization regime.
        # Previously: 20 samples * 500 iters = 25x ratio → catastrophic memorization.
        orchestrator = MLXSelfTrainingOrchestrator(
            base_model_path=base_model,
            iterations=3,
            samples_per_iteration=100,
            generator_type="agentic",
            training_iters=200,
            resume_adapter_path=args.resume
        )
        orchestrator.run(teacher_paths=teacher_paths)

