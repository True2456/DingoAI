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
        learning_rate: float = 1.5e-6,  # 1.5e-6: ultra-stable for MoE models where routing gates are sensitive
        iters: int = 100,
        batch_size: int = 1,
        adapter_path: Optional[str] = None
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        # 1.5e-6: empirically safer for MoE models to prevent router and expert collapse
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
                    # We wrap prompts in the model's native chat template to avoid out-of-distribution shocks.
                    
                    # Sub-step 1: Learn to reason and act.
                    p1_text = f"Task: {sample.get('instruction', '')}"
                    if hasattr(tokenizer, "apply_chat_template"):
                        try:
                            prompt_1 = tokenizer.apply_chat_template(
                                [{"role": "user", "content": p1_text}],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        except Exception:
                            prompt_1 = f"Task: {sample.get('instruction', '')}\n"
                    else:
                        prompt_1 = f"Task: {sample.get('instruction', '')}\n"

                    completion_1 = (
                        f"[THOUGHT]{sample['thought']}[END]\n"
                        f"[ACTION]{sample['actions']}[END]\n"
                    )
                    f.write(json.dumps({"prompt": prompt_1, "completion": completion_1}) + "\n")

                    # Sub-step 2: Learn to answer based on real observation (Multi-turn chat flow)
                    p2_content = f"[THOUGHT]{sample['thought']}[END]\n[ACTION]{sample['actions']}[END]\n"
                    messages_2 = [
                        {"role": "user", "content": p1_text},
                        {"role": "assistant", "content": p2_content},
                        {"role": "user", "content": f"Observation: {sample['observation']}"}
                    ]
                    if hasattr(tokenizer, "apply_chat_template"):
                        try:
                            prompt_2 = tokenizer.apply_chat_template(
                                messages_2,
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        except Exception:
                            prompt_2 = (
                                f"Task: {sample.get('instruction', '')}\n"
                                f"Thought: {sample['thought']}\n"
                                f"Action: {sample['actions']}\n"
                                f"Observation: {sample['observation']}\n"
                            )
                    else:
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
                    
                    p_text = f"Instruction: {inst}\nInput: {inp}\n"
                    if hasattr(tokenizer, "apply_chat_template"):
                        try:
                            prompt = tokenizer.apply_chat_template(
                                [{"role": "user", "content": p_text}],
                                tokenize=False,
                                add_generation_prompt=True
                            )
                        except Exception:
                            prompt = p_text
                    else:
                        prompt = p_text

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
        # rank=16, alpha=32: standard 2x convention.
        # keys: explicitly restrict LoRA to self-attention projections.
        # This completely avoids updating the MoE router gates and expert Feed-Forward blocks,
        # preventing expert routing collapse and preserving the base model's general intelligence.
        lora_config = {
            "rank": 16,
            "alpha": 32,
            "scale": 1.0,
            "dropout": 0.05,
            "keys": ["self_attn.q_proj", "self_attn.v_proj", "self_attn.k_proj", "self_attn.o_proj"]
        }

        # Setup LoRA weights if starting from scratch.
        # num_layers=16: For a 26B MoE, 8 layers covers very little parameters.
        # 16 layers gives enough gradient signal to actually reshape the model's behavior across experts/routing gates.
        if not self.adapter_path:
            mlx_lm.lora.linear_to_lora_layers(model, num_layers=16, config=lora_config)

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
                "num_layers": 16,
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
