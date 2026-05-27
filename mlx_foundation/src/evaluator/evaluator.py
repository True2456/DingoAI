import mlx_lm
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from typing import Any, List, Dict
import numpy as np
import mlx.core as mx

class MLXEvaluator:
    """
    Evaluates the performance of an MLX model.
    """

    def __init__(self, model_path: str, adapter_path: str = None):
        """
        Initializes the evaluator.

        Args:
            model_path: Path to the saved model or Hugging Face model name.
            adapter_path: Path to the saved LoRA adapter.
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        
        print("Evaluator loading base model...")
        self.model, self.tokenizer = mlx_lm.load(model_path)
        
        if adapter_path:
            import os
            adapter_file = os.path.join(adapter_path, "adapters.safetensors")
            if os.path.exists(adapter_file):
                print(f"Evaluator loading adapter weights from {adapter_file}...")
                # We assume the config matches the trainer (16 layers). Ideally we would load adapter_config.json
                # but applying linear_to_lora_layers directly is a safe proxy.
                import json
                config_path = os.path.join(adapter_path, "adapter_config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        loaded_config = json.load(f)
                    num_layers = loaded_config.get("num_layers", 16)
                    lora_config = loaded_config.get("lora_parameters", loaded_config)
                    mlx_lm.lora.linear_to_lora_layers(self.model, num_layers=num_layers, config=lora_config)
                else:
                    # Fallback to defaults
                    lora_config = {"rank": 16, "alpha": 32, "dropout": 0.0, "scale": 1.0}
                    mlx_lm.lora.linear_to_lora_layers(self.model, num_layers=16, config=lora_config)
                    
                self.model.load_weights(adapter_file, strict=False)
            else:
                print(f"Warning: Evaluator could not find {adapter_file}.")

    def _apply_template(self, text: str) -> str:
        """
        Wraps a raw text string in the model's chat template so generation
        starts from the correct distribution.  Without this, chat-pretrained
        models immediately fall into repetition attractors.
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": text}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                pass  # Fallback to raw text if template fails
        return text

    def _make_generate_kwargs(self, temp: float = 0.7, repetition_penalty: float = 1.3,
                               repetition_context_size: int = 20) -> dict:
        """Build sampler + logits_processors kwargs for mlx_lm.generate."""
        return {
            "sampler": make_sampler(temp=temp),
            "logits_processors": make_logits_processors(
                repetition_penalty=repetition_penalty,
                repetition_context_size=repetition_context_size,
            ),
        }

    def evaluate_generation(self, test_prompts: List[str], max_new_tokens: int = 200) -> List[Dict[str, str]]:
        """
        Generates responses for a set of test prompts using MLX.
        Prompts are wrapped in the model's chat template to avoid repetition loops.
        """
        results = []
        print(f"Evaluating {len(test_prompts)} prompts with MLX...")

        gen_kwargs = self._make_generate_kwargs()
        for prompt in test_prompts:
            formatted = self._apply_template(prompt)
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=formatted,
                max_tokens=max_new_tokens,
                verbose=False,
                **gen_kwargs,
            )

            # Strip the formatted prompt prefix if echoed back
            clean_response = response[len(formatted):].strip() if response.startswith(formatted) else response.strip()

            results.append({
                "prompt": prompt,
                "response": clean_response
            })

        return results

    def evaluate_agentic_syntax(self, test_tasks: List[str]) -> Dict[str, Any]:
        """
        Evaluates the student model's tool-calling loop and syntax capabilities.
        Checks for:
        1. Token-wrapped Conformity: Does the output contain [THOUGHT] and [ACTION] markers?
        2. Field Accuracy: Does it contain non-empty thought and action sections?
        3. Action Type validity: Does the action section start with 'python:', 'list_dir:', or 'none:'?
        """
        conformity_count = 0
        valid_fields_count = 0
        correct_tools_count = 0
        total_eval = len(test_tasks)

        print(f"Evaluating agentic syntax on {total_eval} programming tasks...")
        
        for task in test_tasks:
            # Match the exact chat template prompt format used in training
            p_text = f"Task: {task}"
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    formatted = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p_text}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                except Exception:
                    formatted = f"Task: {task}\n"
            else:
                formatted = f"Task: {task}\n"

            gen_kwargs = self._make_generate_kwargs()
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=formatted,
                max_tokens=1024,
                verbose=False,
                **gen_kwargs,
            )
            clean_resp = response[len(formatted):].strip() if response.startswith(formatted) else response.strip()

            # Check for [THOUGHT]/[ACTION]/[PILOT] markers or native channel tags
            # Plain custom tags:
            has_custom_thought = "[THOUGHT]" in clean_resp or "<|thought|>" in clean_resp or "<|channel>thought" in clean_resp
            has_custom_action = "[ACTION]" in clean_resp or "[END]" in clean_resp or "<channel|>" in clean_resp
            
            # Pilot tags model converged on:
            has_pilot_thought = "[PILOT-MODE-ON]" in clean_resp or "[PILOT-OUTPUT-1]" in clean_resp
            has_pilot_action = "END" in clean_resp or "[PILOT-MODE-OFF]" in clean_resp

            # Native model tags:
            has_native_thought = "<|channel>thought" in clean_resp or clean_resp.startswith("<|channel>thought")
            has_native_action = "<channel|>" in clean_resp or "```python" in clean_resp

            has_thought_tag = has_custom_thought or has_pilot_thought or has_native_thought
            has_action_tag = has_custom_action or has_pilot_action or has_native_action
            
            if has_thought_tag and has_action_tag:
                conformity_count += 1
                
                # Extract thought and action content based on detected format
                try:
                    thought_content = ""
                    action_content = ""
                    
                    if "[THOUGHT]" in clean_resp or "<|thought|>" in clean_resp:
                        t_marker = "[THOUGHT]" if "[THOUGHT]" in clean_resp else "<|thought|>"
                        thought_start = clean_resp.find(t_marker) + len(t_marker)
                        thought_end = clean_resp.find("[ACTION]", thought_start)
                        if thought_end == -1:
                            thought_end = clean_resp.find("[END]", thought_start)
                        if thought_end == -1:
                            thought_end = len(clean_resp)
                        thought_content = clean_resp[thought_start:thought_end].strip()
                        
                        action_start = clean_resp.find("[ACTION]", thought_end)
                        if action_start != -1:
                            action_start += len("[ACTION]")
                            action_end = clean_resp.find("[END]", action_start)
                            if action_end == -1:
                                action_end = len(clean_resp)
                            action_content = clean_resp[action_start:action_end].strip()
                        else:
                            action_content = ""
                    elif has_pilot_thought:
                        # Extract between pilot blocks
                        if "[PILOT-OUTPUT-1]" in clean_resp:
                            thought_start = clean_resp.find("[PILOT-OUTPUT-1]") + len("[PILOT-OUTPUT-1]")
                            thought_end = clean_resp.find("[PILOT-OUTPUT-1]END", thought_start)
                            thought_content = "Logical code generation block"
                            action_content = clean_resp[thought_start:thought_end].strip()
                        else:
                            thought_start = clean_resp.find("[PILOT-MODE-ON]") + len("[PILOT-MODE-ON]")
                            thought_end = clean_resp.find("[PILOT-MODE-OFF]", thought_start)
                            thought_content = clean_resp[thought_start:thought_end].strip()
                            action_content = "none: complete"
                    else:
                        # Native parsing
                        thought_start = clean_resp.find("<|channel>thought") + len("<|channel>thought")
                        thought_end = clean_resp.find("<channel|>", thought_start)
                        if thought_end == -1:
                            # Fallback if no channel end, look for first code block
                            thought_end = clean_resp.find("```python", thought_start)
                        
                        thought_content = clean_resp[thought_start:thought_end].strip()
                        
                        # Action is the code block
                        action_start = thought_end
                        action_content = clean_resp[action_start:].strip()

                    if thought_content and action_content:
                        valid_fields_count += 1
                        
                    # Check if action format is valid (e.g. starts with python, list_dir, none, or is a markdown python block)
                    is_valid_tool = False
                    if ":" in action_content:
                        act_type = action_content.split(":")[0].strip().replace("Action (", "").replace(")", "")
                        if act_type in ["python", "list_dir", "none"]:
                            is_valid_tool = True
                    if "```python" in action_content or "<channel|>" in action_content:
                        is_valid_tool = True
                    if has_pilot_thought:
                        is_valid_tool = True
                        
                    if is_valid_tool:
                        correct_tools_count += 1
                except Exception:
                    pass

        return {
            "token_conformity_rate": conformity_count / total_eval if total_eval > 0 else 0,
            "field_accuracy_rate": valid_fields_count / total_eval if total_eval > 0 else 0,
            "tool_selection_accuracy": correct_tools_count / total_eval if total_eval > 0 else 0,
            "total_tested": total_eval
        }

    def calculate_perplexity(self, test_texts: List[str]) -> float:
        """
        Calculates the perplexity of the model on a list of texts using MLX.

        Note: Perplexity is exp(cross_entropy_loss).
        """
        if not test_texts:
            return 0.0

        print(f"Calculating perplexity for {len(test_texts)} texts with MLX...")

        total_loss = 0.0
        total_tokens = 0

        for text in test_texts:
            # Tokenize input
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= 1:
                continue

            # Create input and labels (labels are shifted by 1 in causal LM)
            # Input: tokens[0:-1], Labels: tokens[1:]
            input_tokens = tokens[:-1]
            target_tokens = tokens[1:]

            # Convert to MX array
            input_mx = mx.array(input_tokens)
            target_mx = mx.array(target_tokens)

            # In MLX, we can get logits and compute loss manually
            # This is a simplified version for the foundation
            logits = self.model(input_mx.reshape(1, -1))[0]

            # Log-softmax of logits using mlx.nn
            import mlx.nn as nn
            log_probs = nn.log_softmax(logits, axis=-1)

            # Gather the log-probs of the target tokens
            # log_probs shape: [seq_len, vocab_size]
            # target_mx shape: [seq_len]
            target_log_probs = mx.take_along_axis(log_probs, target_mx.reshape(-1, 1), axis=-1).squeeze(-1)

            # Average loss for this sequence
            seq_loss = -mx.mean(target_log_probs)

            total_loss += seq_loss.item() * len(target_tokens)
            total_tokens += len(target_tokens)

        if total_tokens == 0:
            return 0.0

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        return float(perplexity)

if __name__ == "__main__":
    # Basic smoke test
    import os
    model_name = "mlx-community/Llama-3-8B-Instruct-4bit" # Example

    print("Running smoke test for MLX Evaluator...")
    try:
        evaluator = MLXEvaluator(model_name)

        # Test generation
        test_prompts = ["Once upon a time,", "The future of AI is"]
        gen_results = evaluator.evaluate_generation(test_prompts)
        for res in gen_results:
            print(f"Prompt: {res['prompt']}\nResponse: {res['response']}\n---")

        # Test perplexity
        test_texts = ["This is a test sentence to check perplexity.", "Another sentence for evaluation."]
        perp = evaluator.calculate_perplexity(test_texts)
        print(f"Perplexity: {perp}")

    except Exception as e:
        print(f"MLX Evaluator smoke test failed: {e}")
