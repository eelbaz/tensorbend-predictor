#!/usr/bin/env python3
"""Collect MoE routing data via vLLM and train expert predictors.

Usage:
    python scripts/collect_and_train.py \
        --model-dir /path/to/model \
        --output predictor.pt \
        --n-gpus 8 \
        --predictor-dim 256 \
        --max-tokens 32
"""
import os, sys, json, time, argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


def load_calibration_prompts(n: int = 30) -> list[str]:
    """Load prompts from stepfun-ai/Step-3.5-Flash-SFT, with fallback."""
    try:
        from huggingface_hub import hf_hub_download
        print("  Loading stepfun-ai/Step-3.5-Flash-SFT...", flush=True)
        path = hf_hub_download(
            "stepfun-ai/Step-3.5-Flash-SFT",
            "json/general/chunk_0.json",
            repo_type="dataset",
        )
        data = json.load(open(path))
        prompts = []
        for entry in data:
            convs = entry.get("conversations", [])
            if convs:
                content = convs[0].get("content", "")
                if len(content) > 50:
                    prompts.append(content)
            if len(prompts) >= n:
                break
        print(f"  Loaded {len(prompts)} prompts", flush=True)
        return prompts
    except Exception as e:
        print(f"  Dataset load failed ({e}), using fallback prompts", flush=True)
        return [
            "Explain the three laws of thermodynamics and their implications for entropy.",
            "Describe CRISPR-Cas9 gene editing in detail.",
            "How do quantum computers use superposition for computational speedup?",
            "Describe transformer architecture: attention, feedforward, layer normalization.",
            "What are the main challenges in building a space elevator?",
        ] * 6


def main():
    parser = argparse.ArgumentParser(description="Collect routing data and train expert predictors")
    parser.add_argument("--model-dir", required=True, help="Path to model directory")
    parser.add_argument("--output", default="predictor.pt", help="Output predictor path")
    parser.add_argument("--n-gpus", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--predictor-dim", type=int, default=256, help="Predictor hidden dimension")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--max-tokens", type=int, default=256, help="Tokens to generate per prompt (captures decode phase)")
    parser.add_argument("--n-prompts", type=int, default=30, help="Number of calibration prompts")
    parser.add_argument("--max-model-len", type=int, default=2048, help="vLLM max model length")
    parser.add_argument("--gpu-mem-util", type=float, default=0.85, help="GPU memory utilization")
    args = parser.parse_args()

    print(f"\n{'='*60}", flush=True)
    print("Expert Predictor Training", flush=True)
    print(f"  model: {args.model_dir}", flush=True)
    print(f"  output: {args.output}", flush=True)
    print(f"  gpus: {args.n_gpus}, dim: {args.predictor_dim}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Phase 1: Load model via vLLM
    print("=== Phase 1: Loading model ===", flush=True)
    t0 = time.monotonic()
    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=args.n_gpus,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"  Loaded in {time.monotonic() - t0:.0f}s", flush=True)

    # Phase 2: Hook MoE gates
    print("\n=== Phase 2: Hooking MoE gates ===", flush=True)
    from predictor.hooks import (
        setup_collection_hooks, activate_hooks, deactivate_hooks,
        reset_routing, save_data,
    )
    results = llm.apply_model(setup_collection_hooks)
    moe_layers = results[0] if results else []
    print(f"  MoE layers: {len(moe_layers)}", flush=True)
    if not moe_layers:
        print("ERROR: No MoE layers found", flush=True)
        sys.exit(1)

    # Get config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    tc = getattr(config, "text_config", config)
    hidden_dim = tc.hidden_size
    num_experts = getattr(tc, "n_routed_experts", 72)
    print(f"  hidden_dim={hidden_dim}, num_experts={num_experts}", flush=True)

    # Phase 3: Calibration
    print("\n=== Phase 3: Calibration ===", flush=True)
    prompts = load_calibration_prompts(args.n_prompts)
    sampling = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    llm.apply_model(activate_hooks)
    for i, prompt in enumerate(prompts):
        llm.apply_model(reset_routing)
        try:
            llm.generate([prompt], sampling)
            print(f"  [{i+1}/{len(prompts)}] {prompt[:45]}... done", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(prompts)}] skip ({e})", flush=True)
    llm.apply_model(deactivate_hooks)

    # Save data to disk from worker
    save_info = llm.apply_model(save_data)
    info = save_info[0] if save_info else {}
    print(f"  Collected: {info.get('total_samples', 0)} samples, {info.get('layers', 0)} layers", flush=True)
    if not info.get("saved"):
        print("ERROR: No data collected", flush=True)
        sys.exit(1)

    # Free GPU
    print("  Freeing GPU memory...", flush=True)
    del llm
    import gc; gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Load data from disk
    raw = torch.load("/dev/shm/gate_data_raw.pt", map_location="cpu")
    training_data = {}
    for pi, li in enumerate(moe_layers):
        if li in raw:
            training_data[pi] = raw[li]
    n_per_layer = len(training_data[0][0]) if training_data else 0
    print(f"  {n_per_layer} samples/layer", flush=True)

    # Phase 4: Train
    print("\n=== Phase 4: Training ===", flush=True)
    from predictor import ExpertPredictorSet, train_predictors
    pset = ExpertPredictorSet(
        num_layers=len(moe_layers),
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        predictor_dim=args.predictor_dim,
        device="cuda:0",
        use_prev_routing=True,
    )
    print(f"  Predictor: {pset.total_params/1e6:.1f}M params ({pset.size_mb:.0f}MB)", flush=True)

    t0 = time.monotonic()
    stats = train_predictors(pset, training_data, epochs=args.epochs)
    print(f"  Trained in {time.monotonic() - t0:.0f}s", flush=True)

    accs = [s["accuracy"] for s in stats.values()]
    print(f"  Accuracy: avg={sum(accs)/len(accs):.1f}% min={min(accs):.1f}% max={max(accs):.1f}%", flush=True)

    # Save
    pset.save(args.output)
    size = Path(args.output).stat().st_size / 1e6
    print(f"\n=== Saved: {args.output} ({size:.0f}MB) ===", flush=True)

    # Metadata
    meta = {
        "model_dir": args.model_dir,
        "num_moe_layers": len(moe_layers),
        "moe_layer_indices": moe_layers,
        "hidden_dim": hidden_dim,
        "num_experts": num_experts,
        "predictor_dim": args.predictor_dim,
        "use_prev_routing": True,
        "avg_accuracy": sum(accs) / len(accs),
        "min_accuracy": min(accs),
        "calibration_samples_per_layer": n_per_layer,
        "max_tokens": args.max_tokens,
    }
    meta_path = Path(args.output).with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
