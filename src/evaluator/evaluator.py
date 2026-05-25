import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

class ModelEvaluator:
    """
    Evaluates the performance of a language model.
    """

    def __init__(self, model_path: str, device: str = None):
        """
        Initializes the evaluator.

        Args:
            model_path: Path to the saved model or Hugging Face model name.
            device: Device to use ("mps", "cpu", "cuda"). Defaults to MPS if available on Mac.
        """
        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Initializing evaluator on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def evaluate_generation(self, test_prompts: List[str], max_new_tokens: int = 50) -> List[Dict[str, str]]:
        """
        Generates responses for a set of test prompts.

        Args:
            test_prompts: A list of prompts to evaluate.
            max_new_tokens: Maximum number of tokens to generate for each prompt.

        Returns:
            A list of dictionaries containing the prompt and the generated response.
        """
        results = []
        print(f"Evaluating {len(test_prompts)} prompts...")

        with torch.no_grad():
            for prompt in test_prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # Decode only the newly generated tokens
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Handle cases where the prompt might be included in the output
                response = generated_text[len(prompt):].strip()

                results.append({
                    "prompt": prompt,
                    "response": response
                })

        return results

    def calculate_perplexity(self, test_texts: List[str]) -> float:
        """
        Calculates the perplexity of the model on a list of texts.
        Note: Perplexity is a measure of how well the probability distribution predicted by the model matches the actual distribution of the data.

        Args:
            test_texts: A list of strings to evaluate.

        Returns:
            The average perplexity across the provided texts.
        """
        if not test_texts:
            return 0.0

        print(f"Calculating perplexity for {len(test_texts)} texts...")
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in test_texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                input_ids = inputs["input_ids"]

                # For CausalLM, labels are the same as input_ids
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss

                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)

        if total_tokens == 0:
            return 0.0

        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return perplexity

if __name__ == "__main__":
    # Basic smoke test
    import os
    # Using a small model for testing
    model_name = "distilgpt2"

    print("Running smoke test for Evaluator...")
    try:
        evaluator = ModelEvaluator(model_name)

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
        print(f"Evaluator smoke test failed: {e}")
