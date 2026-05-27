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
        adapter_path: Optional[str] = None,
        max_seq_length: int = 2048  # Reduced to 2048 (max sequence length in dataset is 2003) to prevent Metal OOM.
    ):
        self.model_path = model_path
        self.output_dir = output_dir
        # 1.5e-6: empirically safer for MoE models to prevent router and expert collapse
        self.learning_rate = learning_rate
        self.iters = iters
        self.batch_size = batch_size
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _prepare_data(self, samples: List[Dict[str, Any]], tokenizer, temp_dir: str) -> str:
        """
        Prepares the data in the JSONL format expected by CacheDataset.
        Implements loss masking by tokenizing segment-by-segment and building 
        (input_ids, labels) where labels for environmental observations are masked with -100.
        """
        data_path = os.path.join(temp_dir, "train.jsonl")
        skipped_count = 0

        with open(data_path, "w") as f:
            for i, sample in enumerate(samples):
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

                    completion_1 = f"<|channel>thought\n{sample['thought']}\n<channel|>{sample['actions']}"

                    # Token length safety check
                    p1_len = len(tokenizer.encode(prompt_1))
                    if p1_len >= self.max_seq_length - 16:
                        print(f"WARNING: Skipping Sub-step 1 for sample {i} (prompt length {p1_len} >= {self.max_seq_length - 16})")
                        skipped_count += 1
                        continue

                    f.write(json.dumps({"prompt": prompt_1, "completion": completion_1}) + "\n")

                    # Sub-step 2: Learn to answer based on real observation (Multi-turn chat flow)
                    p2_content = f"<|channel>thought\n{sample['thought']}\n<channel|>{sample['actions']}"
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

                    completion_2 = sample['output']

                    # Token length safety check
                    p2_len = len(tokenizer.encode(prompt_2))
                    if p2_len >= self.max_seq_length - 16:
                        print(f"WARNING: Skipping Sub-step 2 for sample {i} (prompt length {p2_len} >= {self.max_seq_length - 16})")
                        skipped_count += 1
                        continue

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

                    completion = resp

                    # Token length safety check
                    p_len = len(tokenizer.encode(prompt))
                    if p_len >= self.max_seq_length - 16:
                        print(f"WARNING: Skipping sample {i} (prompt length {p_len} >= {self.max_seq_length - 16})")
                        skipped_count += 1
                        continue

                    f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")

        if skipped_count > 0:
            print(f"Total sub-steps/samples skipped due to token limit: {skipped_count}")

        return data_path


    def train(self, samples: List[Dict[str, Any]]):
        """Executes the LoRA training loop."""
        print(f"Starting MLX LoRA training on {len(samples)} samples...")

        # Set cache limit to 80 GB to force MLX to garbage collect intermediate memory
        # blocks aggressively when they exceed the threshold, while allowing sufficient space
        # for unquantized model weights (~52 GB) and activation caches (~3 GB).
        try:
            import mlx.core as mx
            mx.set_cache_limit(80 * 1024 * 1024 * 1024)
            print("Set MLX cache limit to 80 GB.")
        except Exception as e:
            print(f"Warning: Could not set cache limit: {e}")

        # Load model and tokenizer. We strictly load ONLY the base model first 
        # so it is safely memory-mapped from disk (~52GB). 
        # Passing adapter_path directly into mlx_lm.load causes it to load into active memory, doubling VRAM.
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

        # CRITICAL memory optimization: Freeze all base parameters FIRST
        # BEFORE initializing LoRA layers. This ensures the quantized base weights
        # stay completely frozen, preventing the [QuantizedMatmul::vjp] gradient crash.
        model.freeze()

        # Setup LoRA layers unconditionally so the architecture matches what we're training.
        # mlx_lm automatically sets lora_a and lora_b to requires_grad=True internally.
        mlx_lm.lora.linear_to_lora_layers(model, num_layers=16, config=lora_config)

        # Load previous adapter weights if resuming
        if self.adapter_path:
            adapter_file = os.path.join(self.adapter_path, "adapters.safetensors")
            if os.path.exists(adapter_file):
                print(f"Loading previous adapter weights from {adapter_file}...")
                model.load_weights(adapter_file, strict=False)
            else:
                print(f"Warning: Adapter file not found at {adapter_file}. Starting weights from scratch.")

        # Verify trainable parameter size
        import mlx.utils
        trainable = mlx.utils.tree_flatten(model.trainable_parameters())
        trainable_size = sum(v.size for _, v in trainable)
        print(f"Trainable parameters size: {trainable_size / 1e6:.2f} M ({trainable_size * 2 / 1e9:.2f} GB in bf16)")

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
                max_seq_length=self.max_seq_length,
                grad_checkpoint=True,
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
            del optimizer
            del base_set
            del train_set
            import gc
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except AttributeError:
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
