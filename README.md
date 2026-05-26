# LLM Self-Training Foundation

An automated **knowledge distillation loop** for Apple Silicon, using teacher models to generate agentic coding trajectories and fine-tuning a student model via LoRA — all running locally with [MLX](https://github.com/ml-explore/mlx).

## Overview

This project implements a full **Generation → Training → Evaluation** orchestration loop. Large "teacher" models synthesize agentic coding trajectories (thought → action → observation → output) that are used to fine-tune a smaller, more efficient "student" model. No cloud APIs required — everything runs on-device.

```
Teacher (Qwen3-Coder / Gemma-4-31B)
        │
        │  generates agentic trajectories
        ▼
  Synthetic Dataset (JSONL)
        │
        │  LoRA fine-tuning (MLX)
        ▼
  Student (Gemma-4-26B MoE)
        │
        │  evaluate perplexity + agentic syntax
        ▼
  Next Iteration  ──▶  (repeat N times)
```

## Architecture

| Component | Description |
|---|---|
| `src/generator/` | `EnsembleAgenticTrajectoryGenerator` — uses teacher models to bootstrap 100+ unique coding tasks and generate thought/action/observation traces |
| `src/trainer/` | `MLXTrainer` — LoRA fine-tuning via `mlx_lm`. Loss-masks the prompt and environmental observations so the model only learns to generate thoughts, actions, and outputs |
| `src/evaluator/` | `MLXEvaluator` — checks agentic syntax conformity (`<\|thought\|>` / `<\|action\|>` tags), generation quality, and perplexity. Includes a **collapse gate** that halts the loop if the model degenerates |
| `src/main.py` | `MLXSelfTrainingOrchestrator` — wires everything together and enforces the iter/sample safety ratio |

## Key Design Decisions

### Iteration/Sample Ratio
The number of training iterations must scale with the number of samples. A ratio above ~3× causes the model to memorize exact token positions rather than learning generalizable patterns.

```
✅ Safe:    200 iters / 100 samples = 2.0×
❌ Unsafe:  500 iters /  20 samples = 25.0×  ← causes catastrophic collapse
```

### LoRA Configuration
| Parameter | Value | Rationale |
|---|---|---|
| `num_layers` | 8 | 1 layer covers ~0.01% of a 26B MoE — not enough gradient signal |
| `rank` | 16 | Higher rank gives each layer more expressive bandwidth |
| `alpha` | 32 | Standard 2× convention (`alpha = 2 × rank`) |
| `learning_rate` | 3e-6 | MoE routing gates shift dangerously fast at 1e-5 |

### Chat Template in Evaluation
Gemma-4 is a chat-pretrained model. Feeding it a raw string during evaluation puts it in the middle of its pretraining distribution — it immediately falls into a repetition attractor. The evaluator always wraps prompts with `tokenizer.apply_chat_template()`.

### Collapse Gate
After each iteration, if perplexity exceeds **5,000**, the loop halts immediately with a diagnostic message instead of cascading a broken adapter into the next iteration.

## Models

| Role | Model | Notes |
|---|---|---|
| Student | `gemma-4-26b-a4b-it-oQ8` | Gemma 4 26B MoE, 4-bit quantized |
| Teacher 1 | `Qwen3-Coder-Next-MLX-8bit` | Primary code trajectory generator |
| Teacher 2 | `gemma-4-31b-it-oQ8` | Fallback / ensemble teacher |

Models are loaded from `~/.lmstudio/models/` and are **not** included in this repository.

## Installation

```bash
git clone https://github.com/True2456/LLM-Self-Training-Foundation
cd LLM-Self-Training-Foundation/mlx_foundation

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Smoke test (1 iteration, 1 sample, 15 training steps)
```bash
python src/main.py --mode smoke
```

### Full distillation run (3 iterations, 100 samples, 200 steps each)
```bash
python src/main.py --mode full
```

### Resume from an existing adapter
```bash
python src/main.py --mode full --resume models/mlx_self_training/iteration_2
```

Or use the convenience script:
```bash
./run_resume.sh
```

## Output

Trained LoRA adapters are saved to:
```
models/mlx_self_training/
├── iteration_1/
│   ├── adapters.safetensors      ← final adapter weights
│   ├── adapter_config.json
│   ├── 0000100_adapters.safetensors  ← checkpoints
│   └── 0000200_adapters.safetensors
├── iteration_2/
└── iteration_3/
```

Each adapter is a **LoRA diff** on top of the base student model — the base weights are never modified.

## Agentic Format

The student is trained to produce structured agentic traces using custom control tokens:

```
Task: Write a script that counts files in a directory.

<|thought|>I need to use os.listdir() or os.walk() to count files.<|end|>
<|action|>python: import os; print(len([f for f in os.listdir('.') if os.path.isfile(f)]))<|end|>
<|observation|>7<|end|>
<|output|>There are 7 files in the current directory.<|end|>
```

Loss is masked on the `<|observation|>` block — the model learns to reason and act, not to predict environment outputs.

## License

MIT
