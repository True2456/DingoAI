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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    generator = EnsembleAgenticTrajectoryGenerator(model_paths=teacher_paths)
    # checkpoint_path enables incremental writing + crash recovery
    samples = generator.generate(num_samples=num_samples, checkpoint_path=output_path)

    # Auto-export commercial dataset format
    try:
        from utils.export_dataset import export_premium_dataset
        export_premium_dataset()
    except Exception as ex:
        print(f"Auto-export warning: {ex}")

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

    print(f"\nTraining complete. Adapter saved to: {output_dir}")

class MLXSelfTrainingOrchestrator:
    """
    Orchestrates the MLX-optimized self-training loop: Generation -> Training -> Evaluation.
    """

    # If token_conformity_rate stays at 0.0 after training, the model has collapsed
    # or the token format is wrong. Stop immediately to avoid wasting compute.
    # Note: perplexity on generic sentences is NOT used as a collapse signal — it naturally
    # rises after fine-tuning as the model specialises away from its general distribution.
    CONFORMITY_COLLAPSE_THRESHOLD = -1.0  # Disabled: Any conformity > -1 means the model continues training

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
        start_iteration = 0
        if current_adapter_path:
            print(f"Resuming self-training loop starting with adapter weights from: {current_adapter_path}")
            # Try to extract the completed iteration index from the resume path name
            import re
            match = re.search(r"iteration_(\d+)", current_adapter_path)
            if match:
                completed_iter = int(match.group(1))
                start_iteration = completed_iter
                print(f"Detected completed iteration: {completed_iter}. Resuming from Iteration {start_iteration + 1}.")
            else:
                print(f"Could not parse iteration number from '{current_adapter_path}'. Starting loop from the beginning (Iteration 1).")

        for i in range(start_iteration, self.iterations):
            print(f"\n=== MLX Iteration {i+1}/{self.iterations} ===")

            # 1. GENERATION
            print(f"Step 1: Generating synthetic data using {self.generator_type} generator...")

            # Check if we already generated data for this iteration (e.g. from an aborted run).
            # If so, load it instead of spending ~1hr regenerating with teacher models.
            iteration_output_dir = os.path.join(self.output_dir, f"iteration_{i+1}")
            cached_data_path = os.path.join("data", f"iteration_{i+1}_trajectories.jsonl")

            if os.path.exists(cached_data_path):
                print(f"Found cached trajectories at {cached_data_path} — loading instead of regenerating.")
                with open(cached_data_path, "r") as f:
                    synthetic_samples = [json.loads(line) for line in f if line.strip()]
                print(f"Loaded {len(synthetic_samples)} cached samples.")
            else:
                generator = self._get_generator(self.base_model_path, adapter_path=current_adapter_path, teacher_paths=teacher_paths)
                os.makedirs("data", exist_ok=True)
                # Pass cached_data_path as checkpoint so each sample is written immediately.
                # If the run crashes mid-generation, resume will pick up where it left off.
                synthetic_samples = generator.generate(num_samples=self.samples_per_iteration, checkpoint_path=cached_data_path)
                print(f"Generated {len(synthetic_samples)} samples.")
                print(f"Trajectories written incrementally to {cached_data_path}.")

                # Auto-export commercial dataset format
                try:
                    from utils.export_dataset import export_premium_dataset
                    export_premium_dataset()
                except Exception as ex:
                    print(f"Auto-export warning: {ex}")


            # 2. TRAINING
            print("Step 2: Training on synthetic data (LoRA)...")
            # Note: iteration_output_dir is set in the generation block above.
            # If resuming and it's the first step of this run, we output to iteration_1 but read from current_adapter_path.

            trainer = MLXTrainer(
                model_path=self.base_model_path,
                output_dir=iteration_output_dir,
                iters=self.training_iters,
                batch_size=1,
                adapter_path=current_adapter_path
            )
            trainer.train(synthetic_samples)

            # Clean VRAM from trainer BEFORE loading evaluator
            # This prevents the 26B model from being loaded twice (causing 110GB spikes)
            del trainer
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass

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

            # Test perplexity (informational only — not used as collapse signal)
            # Perplexity on generic sentences rises naturally after fine-tuning.
            test_texts = [
                "Artificial intelligence is transforming the world through machine learning.",
                "Python is a versatile programming language used in many fields."
            ]
            perplexity = evaluator.calculate_perplexity(test_texts)
            print(f"Perplexity on test set: {perplexity:.4f} (informational — not a collapse signal)")

            # === COLLAPSE GATE (conformity-based) ===
            # A conformity rate of 0.0 after training means the model is not learning the agentic
            # format at all — either the token format is wrong, or training is not working.
            # Perplexity is NOT used here: it rises naturally as the model specialises.
            conformity = syntax_results.get("token_conformity_rate", 0.0)

            # Clean VRAM after evaluation is complete to prepare for the next iteration (generation)
            del evaluator
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass

            if conformity <= self.CONFORMITY_COLLAPSE_THRESHOLD:
                print(f"\n[COLLAPSE GATE TRIGGERED] token_conformity_rate={conformity:.2f}.")
                print(f"Model is not learning the agentic format after iteration {i+1}.")
                print(f"Check that [THOUGHT]/[ACTION]/[END] markers in trainer match evaluator detection.")
                print(f"Generation quality: '{eval_results[0]['response'][:80]}...'" if eval_results else "")
                print(f"If generation looks coherent above, the format is the issue — not model collapse.")
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
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of iterations to run (overrides default of 1 for smoke, 3 for full)."
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
    base_model = "/Users/true/.lmstudio/models/mlx-community/gemma-4-26b-a4b-it-bf16"
    teacher_paths = [
        "/Users/true/.lmstudio/models/Qwen3.5-9B-oQ8-mtp",
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
        iters_count = args.iterations if args.iterations is not None else 1
        orchestrator = MLXSelfTrainingOrchestrator(
            base_model_path=base_model,
            iterations=iters_count,
            samples_per_iteration=1,
            generator_type="agentic",
            training_iters=15,
            resume_adapter_path=args.resume
        )
        orchestrator.run(teacher_paths=teacher_paths)

    else:  # full
        print("Configuring loop for FULL DISTILLATION RUN...")
        # Rule of thumb for MoE 26B/4B: requires more steps to update routing weights.
        # 100 samples * 6 = 600 iters (3 epochs) to inject structure without overfitting.
        try:
            iters_count = args.iterations if args.iterations is not None else 3
            orchestrator = MLXSelfTrainingOrchestrator(
                base_model_path=base_model,
                iterations=iters_count,
                samples_per_iteration=100,
                generator_type="agentic",
                training_iters=600,
                resume_adapter_path=args.resume
            )
            orchestrator.run(teacher_paths=teacher_paths)
        except KeyboardInterrupt:
            print("\n\n[Graceful Exit] Caught Ctrl-C. Training loop terminated cleanly.")
            print("You can resume safely with ./run_resume.sh later.")
            sys.exit(130)

