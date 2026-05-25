import random
from abc import ABC, abstractmethod
from typing import List, Dict
import torch
from transformers import pipeline

class BaseGenerator(ABC):
    """Base class for all synthetic data generators."""

    @abstractmethod
    def generate(self, num_samples: int) -> List[Dict[str, str]]:
        """Generates a list of synthetic data samples."""
        pass

class TemplateGenerator(BaseGenerator):
    """A simple generator that uses templates to create instruction-response pairs."""

    def __init__(self, templates: List[Dict[str, List[str]]]):
        """
        Initializes the template generator.

        Args:
            templates: A list of dictionaries where each dictionary contains
                       'instruction_templates' and 'response_templates'.
        """
        self.templates = templates

    def generate(self, num_samples: int) -> List[Dict[str, str]]:
        """Generates samples using the provided templates."""
        samples = []
        for _ in range(num_samples):
            template_set = random.choice(self.templates)
            instruction = random.choice(template_set['instruction_templates'])
            response = random.choice(template_set['response_templates'])
            samples.append({
                "instruction": instruction,
                "input": "",
                "output": response
            })
        return samples

class ModelBasedGenerator(BaseGenerator):
    """A generator that uses a pre-trained model to generate synthetic data."""

    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initializes the model-based generator.

        Args:
            model_name: The name of the Hugging Face model to use.
            device: The device to use for generation ("cpu", "mps", or "cuda").
        """
        self.device = device
        # Use text-generation pipeline for simplicity
        self.generator = pipeline("text-generation", model=model_name, device=device)

    def generate(self, num_samples: int, prompts: List[str] = None) -> List[Dict[str, str]]:
        """
        Generates samples based on provided prompts.

        Args:
            num_samples: Number of samples to generate.
            prompts: A list of prompts to use for generation. If None, it will
                     use a default set of prompts.
        """
        if prompts is None:
            prompts = [
                "Write a short story about a robot learning to paint.",
                "Explain how photosynthesis works in simple terms.",
                "What are the benefits of regular exercise?",
                "Describe the plot of Romeo and Juliet."
            ]

        # Ensure we don't exceed the number of prompts if num_samples is larger
        if len(prompts) == 0:
             return []

        samples = []
        for i in range(num_samples):
            prompt = prompts[i % len(prompts)]

            # Generate text
            # max_new_tokens is used to control the length of the response
            result = self.generator(prompt, max_new_tokens=100, do_sample=True, temperature=0.7)
            generated_text = result[0]['generated_text']

            # In a real self-training scenario, we'd want to separate the prompt from the response
            # For this foundation, we'll treat the whole thing as the output for simplicity,
            # or try to split it if the prompt is included.
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text.strip()

            samples.append({
                "instruction": prompt,
                "input": "",
                "output": response
            })
        return samples

if __name__ == "__main__":
    # Test TemplateGenerator
    print("Testing TemplateGenerator...")
    test_templates = [
        {
            "instruction_templates": ["Explain {topic}."],
            "response_templates": ["{topic} is interesting."]
        }
    ]
    tg = TemplateGenerator(test_templates)
    print(tg.generate(2))

    # Test ModelBasedGenerator (Requires internet and model download)
    # Using a small model for testing purposes
    print("\nTesting ModelBasedGenerator (this may take a moment)...")
    try:
        # Using GPT-2 as a lightweight placeholder
        mbg = ModelBasedGenerator(model_name="gpt2", device="cpu")
        print(mbg.generate(2, prompts=["The capital of France is"]))
    except Exception as e:
        print(f"ModelBasedGenerator test failed: {e}")
