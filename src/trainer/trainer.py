import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from typing import List, Dict, Any

class SelfTrainingTrainer:
    """
    A trainer wrapper for fine-tuning LLMs using synthetic data.
    Optimized for MPS (Metal Performance Shaders) on macOS.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: str = "models/output",
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        num_epochs: int = 3,
        max_length: int = 512
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.max_length = max_length

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def _prepare_dataset(self, samples: List[Dict[str, str]]) -> Dataset:
        """Converts raw samples into a Hugging Face Dataset."""

        def tokenize_function(examples):
            # Format for causal language modeling: instruction + input + response
            # For simplicity in this foundation, we combine instruction and output
            texts = [
                f"Instruction: {inst}\nInput: {inp}\nResponse: {resp}{self.tokenizer.eos_token}"
                for inst, inp, resp in zip(examples['instruction'], examples['input'], examples['output'])
            ]
            tokenized = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            # For CausalLM, labels are usually the input_ids shifted
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        # Create dataset from list of dicts
        dataset = Dataset.from_list(samples)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        return tokenized_dataset

    def train(self, samples: List[Dict[str, str]]):
        """Executes the training loop."""
        print(f"Preparing dataset with {len(samples)} samples...")
        train_dataset = self._prepare_dataset(samples)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_strategy="no",
            save_strategy="epoch",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )

        print("Starting training...")
        trainer.train()
        print(f"Training complete. Model saved to {self.output_dir}")

        # Save the final model and tokenizer
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

if __name__ == "__main__":
    # Basic test
    import os
    test_samples = [
        {"instruction": "What is AI?", "input": "", "output": "AI is artificial intelligence."},
        {"instruction": "Who are you?", "input": "", "output": "I am a language model."},
    ]

    # Using a tiny model for a quick smoke test
    # Note: This will download files
    trainer = SelfTrainingTrainer(
        model_name="distilgpt2",
        output_dir="models/test_model",
        num_epochs=1,
        batch_size=1
    )
    trainer.train(test_samples)
