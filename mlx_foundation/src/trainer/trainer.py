import os
import json
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path
import types
import mlx_lm
import mlx_lm.lora
from mlx_lm.tuner.datasets import load_local_dataset
import mlx.core as mx
import mlx.optimizers as optim

class MLXTrainer:
    """
    An MLX-optimized trainer using LoRA for efficient fine-tuning.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "models/mlx_output",
        learning_rate: float = 3e-6,  # 3e-6 is empirically safer for MoE; 1e-5 caused collapse
        iters: int = 100,
        batch_size: int = 1,
        adapter_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        # 3e-6: empirically safer for MoE models where routing gates are sensitive.
        # 1e-5 caused the routing gates to over-specialize on 20 samples in <100 steps.
        self.learning_rate = learning_rate
        self.iters = iters
        self.batch_size = batch_size
        self.adapter_path = adapter_path

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _prepare_data(self, samples: List[Dict[str, Any]], tokenizer, temp_dir: str) -> str:
        """
        Prepares the data in the JSONL format expected by CacheDataset.
        Implements loss masking by tokenizing segment-by-segment and building 
        (input_ids, labels) where labels for environmental observations are masked with -100.
        """
        # Note: Since the output format is parsed via TextDataset inside mlx_lm,
        # TextDataset.process returns (tokens, 0) where the second element is the prompt mask length.
        # mlx_lm supports masking the entire prompt (which is everything before the output).
        # However, for advanced agentic masking: we want the model to learn to generate Thoughts and Actions,
        # but NOT learn to generate the environmental Observations.
        # To do this inside mlx_lm's standard TextDataset pipeline: we can set the "mask_prompt" parameter.
        # In mlx_lm, CompletionsDataset supports a precise split of prompt and completion where the prompt is masked.
        # Let's write the dataset in the Completions format!
        # Standard Completions format expected by CompletionsDataset: {"prompt": "...", "completion": "..."}
        # In this format, we package all the history (including previous thoughts/actions/observations)
        # into the "prompt", and the NEXT Thought + Action or Final Answer into the "completion".
        # This naturally masks out all previous outputs and observations perfectly!
        
        data_path = os.path.join(temp_dir, "train.jsonl")

        with open(data_path, "w") as f:
            for sample in samples:
                if "thought" in sample:
                    # Multi-turn parsing to split history (prompt) from the target action (completion)
                    # For a single-step trajectory trace in this smoke test:
                    # prompt is: "Task: <instruction>"
                    # completion is: "<|thought|><thought><|end|>\n<|action|><actions><|end|>\n<|observation|><obs><|end|>\n<|output|><output><|end|>"
                    # But we don't want the model to learn to output the observation!
                    # So we split the completion right before the observation.
                    # First sub-step: prompt = task; completion = thought + action.
                    # Second sub-step: prompt = task + thought + action + observation; completion = final output.
                    
                    # Sub-step 1: Learn to reason and act.
                    # Using [THOUGHT]/[ACTION]/[END] markers instead of <|thought|> style tags.
                    # Reason: Gemma-4 treats <|...|> as its own special token prefix, causing
                    # the tokenizer to split our tags into <|channel> + thought — corrupting learning.
                    # Plain square-bracket markers tokenize cleanly on every model.
                    prompt_1 = f"Task: {sample.get('instruction', '')}\n"
                    completion_1 = (
                        f"[THOUGHT]{sample['thought']}[END]\n"
                        f"[ACTION]{sample['actions']}[END]\n"
                    )
                    f.write(json.dumps({"prompt": prompt_1, "completion": completion_1}) + "\n")

                    # Sub-step 2: Learn to answer based on real observation
                    prompt_2 = (
                        f"Task: {sample.get('instruction', '')}\n"
                        f"Thought: {sample['thought']}\n"
                        f"Action: {sample['actions']}\n"
                        f"Observation: {sample['observation']}\n"
                    )
                    completion_2 = f"[OUTPUT]{sample['output']}[END]"
                    f.write(json.dumps({"prompt": prompt_2, "completion": completion_2}) + "\n")
                else:
                    # Standard completion format
                    inst = sample.get("instruction", "")
                    inp = sample.get("input", "")
                    resp = sample.get("output", "")
                    
                    prompt = f"Instruction: {inst}\nInput: {inp}\n"
                    completion = f"Response: {resp}"
                    f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

        return data_path


    def train(self, samples: List[Dict[str, Any]]):
        """Executes the LoRA training loop."""
        print(f"Starting MLX LoRA training on {len(samples)} samples...")

        # Load model and tokenizer to get parameters for optimizer
        if self.adapter_path:
            print(f"Loading base model with previous adapter from {self.adapter_path}...")
            model, tokenizer = mlx_lm.load(self.model_path, adapter_path=self.adapter_path)
        else:
            print("Loading base model...")
            model, tokenizer = mlx_lm.load(self.model_path)

        # LoRA config:
        # rank=16, alpha=32: standard 2x convention. Higher rank gives each layer more
        # expressive bandwidth — important because we're only training 8 out of ~70+ layers.
        lora_config = {
            "rank": 16,
            "alpha": 32,
            "scale": 1.0,
            "dropout": 0.05,
        }

        # Setup LoRA weights if starting from scratch.
        # num_layers=8: For a 26B MoE, 1 LoRA layer covers ~0.01% of parameters.
        # 8 layers gives enough gradient signal to actually reshape the model's behavior.
        if not self.adapter_path:
            mlx_lm.lora.linear_to_lora_layers(model, num_layers=8, config=lora_config)

        # Now create optimizer only for the trainable parameters
        optimizer = optim.Adam(learning_rate=self.learning_rate)

        with tempfile.TemporaryDirectory() as temp_dir:
            train_data_path = self._prepare_data(samples, tokenizer, temp_dir)

            # Prepare TrainingArgs (excluding learning_rate as it's in the optimizer)
            args = mlx_lm.lora.TrainingArgs(
                batch_size=self.batch_size,
                iters=self.iters,
                adapter_file=os.path.join(self.output_dir, "adapters.safetensors"),
                steps_per_report=10,
                steps_per_eval=100,
            )

            # Load the dataset object from the path. CompletionsDataset wraps the list of dicts with 'prompt' and 'completion' keys
            # and automatically supports loss-masking prompts when mask_prompt=True.
            base_set = mlx_lm.tuner.datasets.CompletionsDataset(
                [json.loads(line) for line in open(train_data_path, "r")],
                tokenizer,
                prompt_key="prompt",
                completion_key="completion",
                mask_prompt=True
            )
            train_set = mlx_lm.tuner.datasets.CacheDataset(base_set)

            # mlx_lm.lora.train requires: model, optimizer, train_dataset
            mlx_lm.lora.train(
                model=model,
                optimizer=optimizer,
                train_dataset=train_set,
                args=args
            )

            # Clean VRAM to prevent memory leak before evaluation
            del model
            del tokenizer
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except ImportError:
                pass

        # Save the adapter configuration so that load_adapters works correctly
        config_path = os.path.join(self.output_dir, "adapter_config.json")
        with open(config_path, "w") as f:
            json.dump({
                "num_layers": 8,
                "lora_parameters": lora_config,
                "fine_tune_type": "lora"
            }, f, indent=4)

        print(f"Training complete. Adapter and config saved to {self.output_dir}")

if __name__ == "__main__":
    # Basic test
    test_samples = [
        {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
        {"instruction": "Who are you?", "input": "", "output": "I am a language model."},
    ]

    # Note: This requires a valid MLX model path
    print("Please provide a valid MLX model path to run this test.")
