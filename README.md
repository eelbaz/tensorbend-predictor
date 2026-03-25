# tensorbend-predictor

Cross-layer expert predictor for Mixture-of-Experts models. Predicts which experts the router will select **before** the gate runs, enabling prefetch of expert weights from NVMe/DRAM or pre-staging of EP all2all buffers.

Trains a lightweight MLP per MoE layer that takes the pre-MoE hidden states + previous layer's routing decision as input, achieving 99.8% training accuracy and 96.9% live prefill accuracy on DeepSeek-style MoE architectures.

## Step-by-step reproduction

These instructions reproduce the validated results on `0xSero/Kimi-K2.5-PRISM-REAP-72` with 8x H100.

### 1. Setup

```bash
git clone https://github.com/eelbaz/tensorbend-predictor.git
cd tensorbend-predictor
pip install -r requirements.txt
```

Verify vLLM can load the model:
```bash
vllm serve 0xSero/Kimi-K2.5-PRISM-REAP-72 \
    --tensor-parallel-size 8 \
    --max-model-len 2048 \
    --trust-remote-code \
    --port 8000 &

# Wait for "Application startup complete", then test:
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"0xSero/Kimi-K2.5-PRISM-REAP-72","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}'

# Kill the server after testing:
pkill -f "vllm serve"
```

### 2. Download the model (if not using HF hub directly)

```bash
huggingface-cli download 0xSero/Kimi-K2.5-PRISM-REAP-72 \
    --local-dir /path/to/kimi-reap-72
```

### 3. Train the predictor

```bash
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
python scripts/collect_and_train.py \
    --model-dir /path/to/kimi-reap-72 \
    --output predictor_reap72.pt \
    --n-gpus 8 \
    --predictor-dim 256
```

This takes ~25 minutes:
- Phase 1 (~3 min): loads model via vLLM offline inference
- Phase 2 (~1 sec): installs gate forward hooks
- Phase 3 (~20 min): runs 100 calibration prompts, collects routing data
- Phase 4 (~2 min): trains 60 per-layer predictors

Expected output:
```
Accuracy: avg=99.8% min=99.4% max=100.0%
Saved: predictor_reap72.pt (450MB)
```

### 4. Verify with live vLLM inference

The repo includes a pre-trained predictor (`predictor_reap72_dim256.pt`) you can use directly:

```python
import torch
from predictor import ExpertPredictorSet

# Load predictor
pset = ExpertPredictorSet(
    num_layers=60, hidden_dim=7168, num_experts=72,
    predictor_dim=256, device="cuda:0", use_prev_routing=True,
)
pset.load("predictor_reap72_dim256.pt")

# Predict experts for a given hidden state
h = torch.randn(1, 7168)  # pre-MoE hidden state
predicted_experts = pset.predict(layer_idx=0, hidden_states=h, top_k=8)
print(f"Predicted experts: {predicted_experts}")
```

## Adapting to other models

The predictor works with any DeepSeek-style MoE model served by vLLM. Key parameters to adjust:

| Parameter | REAP-72 | REAP-530B | Step-3.5-Flash |
|-----------|---------|-----------|----------------|
| `--n-gpus` | 8 | 8 | 4-8 |
| `num_experts` | 72 | 192 | 288 |
| `hidden_dim` | 7168 | 7168 | 4096 |
| MoE layers | 60 | 60 | 42 |

These are auto-detected from the model config. No manual changes needed.

```bash
# REAP-530B (192 experts)
python scripts/collect_and_train.py \
    --model-dir Ex0bit/Kimi-K2.5-PRISM-REAP-530B-A32B \
    --output predictor_reap530b.pt \
    --n-gpus 8
```

## GPU configurations

| GPU | VRAM | Flags | Notes |
|-----|------|-------|-------|
| RTX 3090 (24GB) | `--n-gpus 1 --max-model-len 512 --gpu-mem-util 0.92` | Small REAP models only |
| RTX 4090 (24GB) | `--n-gpus 1 --max-model-len 1024 --gpu-mem-util 0.90` | Small REAP models only |
| 2x RTX 4090 | `--n-gpus 2 --max-model-len 2048` | Medium models |
| 4x A100 (80GB) | `--n-gpus 4 --max-model-len 4096` | Most models |
| 8x H100 (80GB) | `--n-gpus 8 --max-model-len 4096` | All models |

## Choosing predictor_dim

| Dim | Params/layer | Total (60 layers) | Size | Accuracy |
|-----|-------------|-------------------|------|----------|
| 64 | 463K | 28M | 112MB | ~80% |
| 128 | 926K | 56M | 225MB | ~89% |
| **256** | **1.8M** | **112M** | **449MB** | **~99%** |
| 512 | 3.7M | 225M | 899MB | ~99% |

`--predictor-dim 256` is the recommended default.

## Validated results

Trained and live-validated on `0xSero/Kimi-K2.5-PRISM-REAP-72` (8x H100 80GB):

| Phase | Accuracy |
|-------|----------|
| Training | 99.8% avg (min 99.4%) |
| Live prefill | 96.9% |
| Live decode (32 tok) | ~69% |

Key findings:
- **Pure BCE** outperforms ranking loss + importance weighting (99.8% vs 86%)
- **100 diverse prompts** > 30 prompts with longer generation (generalization matters)
- **Prefill accuracy** tracks training accuracy; decode is an architectural ceiling for feedforward predictors

## Architecture

```
predictor/
    __init__.py          # Public API
    model.py             # ExpertPredictor, ExpertPredictorSet
    train.py             # Training (pure BCE, cosine LR)
    hooks.py             # vLLM gate hook for data collection
scripts/
    collect_and_train.py # End-to-end training
paper.pdf                # Technical report
predictor_reap72_dim256.pt  # Pre-trained weights (Git LFS)
```

## How the hooks work

A single gate forward hook captures both:
1. **Gate input** = exact hidden states the gate kernel receives
2. **Gate output** = exact router logits from the kernel

This guarantees the predictor trains on the same data the gate uses, with zero numerical mismatch. The hooks use vLLM's `apply_model()` API and only collect on TP rank 0.

## Paper

See `paper.pdf` for the full technical report including:
- Architecture details and design decisions
- Training methodology and loss function ablation
- End-to-end performance on DGX Spark (1.77 tok/s, 9.5 GB GPU)
- Comparison to 15+ related works (2024-2026)

## License

Apache-2.0
