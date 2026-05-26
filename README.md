# LLM Self-Training Foundation

An automated **knowledge distillation loop** for Apple Silicon, using teacher models to generate agentic coding trajectories and fine-tuning a student model via LoRA — all running locally with [MLX](https://github.com/ml-explore/mlx).

## Flow Diagram

```mermaid
flowchart TD
    subgraph GENERATE["⚙️ Step 1 — Generation (Teacher Machine)"]
        A["DynamicTaskGenerator\nBootstraps 100 unique coding tasks\nvia Qwen3-Coder teacher"] --> B["EnsembleAgenticTrajectoryGenerator\nAlternates between teacher models\nGenerates thought → action → obs traces"]
        B --> C{"Sandbox\nExecutor"}
        C -->|"✅ Code ran correctly"| D["Accept trajectory"]
        C -->|"❌ Execution failed"| E["Retry with alternate\nteacher (up to 3×)"]
        E --> B
        D --> F[("data/iteration_N_trajectories.jsonl\n💾 Persisted — never regenerated")]
    end

    subgraph TRAIN["🧠 Step 2 — Training (Primary Machine)"]
        F --> G["MLXTrainer\nLoRA fine-tuning\n8 layers · rank 16 · LR 3e-6"]
        G --> H["Loss masking:\nLearn [THOUGHT] + [ACTION] + [OUTPUT]\nMask [OBS] (env output — not learnable)"]
        H --> I[("models/mlx_self_training/\niteration_N/adapters.safetensors")]
    end

    subgraph EVAL["📊 Step 3 — Evaluation"]
        I --> J["MLXEvaluator\nLoads student + adapter"]
        J --> K["Agentic syntax check\nDoes output contain\n[THOUGHT][ACTION][END]?"]
        J --> L["Generation quality\napply_chat_template →\nno repetition loops"]
        J --> M["Perplexity\n(informational only —\nnaturally rises after fine-tuning)"]
        K --> N{"Collapse Gate\nconformity_rate > 0?"}
        N -->|"✅ Pass"| O{"More\niterations?"}
        N -->|"❌ Fail — format broken"| P["🛑 Stop loop\nPrint diagnosis"]
        O -->|"Yes"| GENERATE
        O -->|"No"| Q["✅ Done\nFinal adapter ready"]
    end

    subgraph TWOMACHINE["🖥️ Two-Machine Mode (Optional)"]
        R["Second MacBook\nM3 Max 64GB\nTeacher models only"] -->|"./run_generate.sh 100"| S["data/batch_TIMESTAMP.jsonl"]
        S -->|"AirDrop / scp"| T["Primary MacBook\n./run_train_only.sh batch.jsonl"]
        T --> TRAIN
    end
```

## Overview

This project implements a full **Generation → Training → Evaluation** orchestration loop. Large "teacher" models synthesize agentic coding trajectories (thought → action → observation → output) that are used to fine-tune a smaller, more efficient "student" model. No cloud APIs required — everything runs on-device.

## Architecture

| Component | Description |
|---|---|
| `src/generator/generator.py` | `DynamicTaskGenerator` bootstraps unique coding tasks. `EnsembleAgenticTrajectoryGenerator` generates verified thought/action/obs traces using alternating teacher models and a real sandbox executor |
| `src/trainer/trainer.py` | `MLXTrainer` — LoRA fine-tuning via `mlx_lm`. Loss-masks the prompt and `[OBS]` blocks so the model only learns to generate `[THOUGHT]`, `[ACTION]`, and `[OUTPUT]` |
| `src/evaluator/evaluator.py` | `MLXEvaluator` — checks agentic syntax conformity, generation quality (via chat template), and perplexity (informational). Collapse gate uses conformity rate, not perplexity |
| `src/main.py` | `MLXSelfTrainingOrchestrator` — wires everything together, caches generated data, enforces the iter/sample safety ratio |

## Key Design Decisions

### Iteration/Sample Ratio
The number of training iterations must scale with the number of samples. A ratio above ~3× causes the model to memorize exact token positions rather than learning generalizable patterns.

```
✅ Safe:    200 iters / 100 samples = 2.0×
❌ Unsafe:  500 iters /  20 samples = 25.0×  ← causes catastrophic memorization
```

### Agentic Token Format
Custom control tokens use plain square-bracket markers — **not** `<|...|>` style.

```
✅  [THOUGHT]...[END]  [ACTION]...[END]  [OUTPUT]...[END]
❌  <|thought|>        ← Gemma-4 tokenizer treats <|...| > as its own
                          special token prefix, corrupting the learned format
```

### LoRA Configuration
| Parameter | Value | Rationale |
|---|---|---|
| `num_layers` | 8 | 1 layer covers ~0.01% of a 26B MoE — not enough gradient signal |
| `rank` | 16 | Higher rank gives each layer more expressive bandwidth |
| `alpha` | 32 | Standard 2× convention (`alpha = 2 × rank`) |
| `learning_rate` | 3e-6 | MoE routing gates shift dangerously fast at 1e-5 |

### Collapse Gate (Conformity-Based)
After each iteration, if `token_conformity_rate == 0.0` the loop halts. Perplexity is **not** used as a collapse signal — it rises naturally after fine-tuning as the model specialises away from its general distribution.

### Trajectory Caching
Generated trajectories are saved to `data/iteration_N_trajectories.jsonl` immediately after generation. On the next run, if the file exists it's loaded instead of regenerating (~1hr of teacher compute saved per iteration). Delete the file to force fresh generation.

## Two-Machine Workflow

If you have a second Apple Silicon machine (64GB RAM), use it purely for generation — it only needs the teacher models, not the 26B student.

```
Second MacBook (M3 Max 64GB)        Primary MacBook
─────────────────────────           ─────────────────────
./run_generate.sh 100               (training on prev batch)
  → data/batch_TIMESTAMP.jsonl
            │
            │  AirDrop / scp
            ▼
                                    ./run_train_only.sh data/batch.jsonl
```

Generation and training overlap — while your primary machine trains on one batch, the second machine builds the next.

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
cd LLM-Self-Training-Foundation

python -m venv mlx_foundation/venv
source mlx_foundation/venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Smoke test (1 iteration, 1 sample, 15 training steps)
```bash
./run_smoke.sh
```

### Full distillation run (3 iterations, 100 samples, 200 steps each)
```bash
./run_resume.sh          # auto-detects latest checkpoint and resumes, or starts fresh
```

### Generate trajectories only (second machine)
```bash
./run_generate.sh 100                        # generate 100 trajectories
./run_generate.sh 200 data/my_batch.jsonl    # custom count and path
```

### Train on a pre-generated batch (primary machine)
```bash
./run_train_only.sh data/batch.jsonl         # 200 iters (default)
./run_train_only.sh data/batch.jsonl 150     # custom iter count
```

### Manual modes
```bash
python mlx_foundation/src/main.py --mode smoke
python mlx_foundation/src/main.py --mode full
python mlx_foundation/src/main.py --mode generate-only --samples 100 --output data/batch.jsonl
python mlx_foundation/src/main.py --mode train-only --data data/batch.jsonl
```

## Output Structure

```
models/mlx_self_training/
├── iteration_1/
│   ├── adapters.safetensors          ← final adapter weights
│   ├── adapter_config.json
│   ├── 0000100_adapters.safetensors  ← mid-run checkpoints
│   └── 0000200_adapters.safetensors
├── iteration_2/
└── iteration_3/

data/
├── iteration_1_trajectories.jsonl    ← cached — never regenerated
├── iteration_2_trajectories.jsonl
└── iteration_3_trajectories.jsonl
```

Each adapter is a **LoRA diff** on top of the base student model — the base weights are never modified.

## Agentic Format

The student is trained to produce structured agentic traces using plain-text control markers:

```
Task: Write a script that counts files in a directory.

[THOUGHT]I need to use os.listdir() to count files in the current directory.[END]
[ACTION]python: import os; print(len([f for f in os.listdir('.') if os.path.isfile(f)]))[END]
[OBS]7[END]
[OUTPUT]There are 7 files in the current directory.[END]
```

Loss is masked on the `[OBS]` block — the model learns to reason and act, not to predict environment outputs.

## License

MIT
