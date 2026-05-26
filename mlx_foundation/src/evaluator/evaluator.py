import mlx_lm
from typing import List, Dict
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
        # mlx_lm.load returns model and tokenizer
        if adapter_path:
            print(f"Evaluator loading model with adapter from {adapter_path}...")
            self.model, self.tokenizer = mlx_lm.load(model_path, adapter_path=adapter_path)
        else:
            self.model, self.tokenizer = mlx_lm.load(model_path)

    def evaluate_generation(self, test_prompts: List[str], max_new_tokens: int = 150) -> List[Dict[str, str]]:
        """
        Generates responses for a set of test prompts using MLX.
        """
        results = []
        print(f"Evaluating {len(test_prompts)} prompts with MLX...")

        for prompt in test_prompts:
            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False
            )

            # Clean response (remove prompt if it was returned)
            clean_response = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()

            results.append({
                "prompt": prompt,
                "response": clean_response
            })

        return results

    def evaluate_agentic_syntax(self, test_tasks: List[str]) -> Dict[str, Any]:
        """
        Evaluates the student model's tool-calling loop and syntax capabilities.
        Checks for:
        1. Token-wrapped Conformity: Does the generated action contain <|thought|> and <|action|> tags?
        2. Field Accuracy: Does it contain non-empty thought and action sections?
        3. Action Type validity: Does it select a valid action_type ('python', 'list_dir', or 'none')?
        """
        conformity_count = 0
        valid_fields_count = 0
        correct_tools_count = 0
        total_eval = len(test_tasks)

        print(f"Evaluating agentic syntax on {total_eval} programming tasks...")
        
        for task in test_tasks:
            # Match the exact training prompt format
            prompt = f"Task: {task}\n"

            response = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=256,
                verbose=False
            )
            clean_resp = response[len(prompt):].strip() if response.startswith(prompt) else response.strip()

            # Check for <|thought|> and <|action|> tags
            has_thought_tag = "<|thought|>" in clean_resp and "<|end|>" in clean_resp
            has_action_tag = "<|action|>" in clean_resp
            
            if has_thought_tag and has_action_tag:
                conformity_count += 1
                
                # Extract thought and action content
                try:
                    thought_start = clean_resp.find("<|thought|>") + len("<|thought|>")
                    thought_end = clean_resp.find("<|end|>", thought_start)
                    thought_content = clean_resp[thought_start:thought_end].strip()
                    
                    action_start = clean_resp.find("<|action|>", thought_end) + len("<|action|>")
                    action_end = clean_resp.find("<|end|>", action_start)
                    action_content = clean_resp[action_start:action_end].strip()
                    
                    if thought_content and action_content:
                        valid_fields_count += 1
                        
                    # Check if action format is valid (e.g., starts with 'python:' or 'list_dir:' or 'none:')
                    if ":" in action_content:
                        act_type = action_content.split(":")[0].strip()
                        if act_type in ["python", "list_dir", "none"]:
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
