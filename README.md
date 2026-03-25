# tensorbend-predictor

Cross-layer expert predictor for Mixture-of-Experts models. Predicts which experts the router will select **before** the gate runs, enabling prefetch of expert weights from NVMe/DRAM or pre-staging of EP all2all buffers.

Trains a lightweight MLP per MoE layer that takes the pre-MoE hidden states + previous layer's routing decision as input, achieving 96%+ top-8 routing overlap on DeepSeek-style MoE architectures.

## How it works

Each MoE layer has a gate that selects top-k experts based on hidden states. The predictor learns to approximate this gate using:

1. **Hidden states before MoE** — the same signal the gate uses
2. **Previous layer's routing** — exploits 50-60% cross-layer expert overlap

This runs in <0.05ms per layer on GPU, fast enough to hide expert loading latency during the attention phase.

## Quick start

### Prerequisites

- NVIDIA GPU with CUDA 12+
- Python 3.10+
- A Kimi/DeepSeek MoE model (e.g. `0xSero/Kimi-K2.5-PRISM-REAP-72`)

### Install

```bash
pip install -r requirements.txt
```

### Train a predictor

```bash
# Single GPU (RTX 3090/4090) — small models or with reduced context
python scripts/collect_and_train.py \
    --model-dir /path/to/model \
    --output predictor.pt \
    --n-gpus 1 \
    --predictor-dim 256 \
    --max-model-len 1024 \
    --gpu-mem-util 0.90

# Multi-GPU (8x H100) — large models
python scripts/collect_and_train.py \
    --model-dir /path/to/model \
    --output predictor.pt \
    --n-gpus 8 \
    --predictor-dim 256 \
    --max-model-len 2048
```

### GPU configurations

| GPU | VRAM | Recommended flags |
|-----|------|-------------------|
| RTX 3090 (24GB) | `--n-gpus 1 --max-model-len 512 --gpu-mem-util 0.92` | Small REAP models only |
| RTX 4090 (24GB) | `--n-gpus 1 --max-model-len 1024 --gpu-mem-util 0.90` | Small REAP models only |
| 2x RTX 4090 | `--n-gpus 2 --max-model-len 2048` | Medium models |
| 8x H100 (80GB) | `--n-gpus 8 --max-model-len 4096` | All models |

### Choosing predictor_dim

| Dim | Params/layer | Total (60 layers) | Size | Typical accuracy |
|-----|-------------|-------------------|------|-----------------|
| 64 | 463K | 28M | 112MB | ~80% |
| 128 | 926K | 56M | 225MB | ~89% |
| 256 | 1.8M | 112M | 449MB | ~96% |
| 512 | 3.7M | 225M | 899MB | ~99% |

`--predictor-dim 256` is the recommended default. Use 128 for memory-constrained setups.

## Tested models

| Model | Experts | Validated |
|-------|---------|-----------|
| `0xSero/Kimi-K2.5-PRISM-REAP-72` | 72 | 96.4% accuracy (dim=256, 8x H100) |
| `Ex0bit/Kimi-K2.5-PRISM-REAP-530B-A32B` | 192 | Architecture compatible (untested on this tool) |

## Architecture

```
predictor/
    __init__.py          # Public API
    model.py             # ExpertPredictor, ExpertPredictorSet
    train.py             # Training loop with importance weighting
    hooks.py             # vLLM model hooks for data collection
scripts/
    collect_and_train.py # End-to-end training script
```

The predictor uses vLLM's `apply_model()` API to hook into the model's MoE layers during inference, capturing hidden states and actual routing decisions. This avoids the need for HF `from_pretrained` loading (which has compatibility issues with compressed-tensors W4A16 quantization).

## Validated results

Trained and validated on `0xSero/Kimi-K2.5-PRISM-REAP-72` with 8x H100 80GB:

- **Prefill accuracy: 96.7%** — predictor matches gate routing on prompt processing
- **Training accuracy: 96.4%** avg, 93.4% min, 98.2% max across 60 MoE layers
- **Predictor size: 449MB** (112M params, dim=256)
- **Training time: ~70s** on single GPU after data collection

## License

Apache-2.0
