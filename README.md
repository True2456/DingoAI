# LLM Self-Training Foundation (MPS Optimized)

A complete software foundation for implementing an automated self-training (self-instruction) loop for Large Language Models, optimized specifically for macOS (Apple Silicon/MPS).

## Overview

This project implements a full orchestration loop: **Generation $\rightarrow$ Training $\rightarrow$ Evaluation**. It is designed to leverage the high memory bandwidth of Apple Silicon to perform knowledge distillation—using a large "Teacher" model to generate high-quality synthetic data to train a smaller, more efficient "Student" model.

## Core Components

- **`src/generator/`**: Implements both template-based and model-based synthetic data generation.
- **`src/trainer/`**: A wrapper around the Hugging Face `Trainer` API, optimized for Metal Performance Shaders (MPS).
- **`src/evaluator/`**: Provides tools for qualitative (text generation) and quantitative (perplexity) model assessment.
- **`src/main.py`**: The orchestrator that manages the iterative self-improvement loop.

## Features

- 🚀 **MPS Acceleration**: Native support for Apple Silicon GPU acceleration via PyTorch.
- 🔄 **Automated Loop**: Full automation from data generation to model evaluation.
- 🧠 **Knowledge Distillation**: Designed to "compress" intelligence from large models into smaller ones.
- 🛠️ **Modular Design**: Easily swap out generators, trainers, or evaluators.

## Installation

1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd LLM
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the integrated smoke test to verify the entire pipeline:

```bash
python src/main.py
```

### Advanced Training

To perform real-world distillation, modify the `SelfTrainingOrchestrator` in `src/main.py` to use a larger base model (e.g., Llama-3-70B) and increase the number of iterations and samples.

## Model Weights

**Note:** This repository contains only the source code. To keep the repository lightweight, model weights are not included.

To use the pipeline, you can download models directly from Hugging Face. For optimal results in distillation, we recommend:
- **Teacher Model:** A large, high-reasoning model (e.g., `meta-llama/Meta-Llama-3-70B`).
- **Student Model:** A smaller, efficient model (e.g., `meta-llama/Meta-Llama-3-8B`).

## License

[Specify License, e.g., MIT]
