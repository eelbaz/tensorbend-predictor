"""vLLM model hooks for routing data collection.

Captures exact gate output (router_logits) as ground truth using a SINGLE
gate forward hook that records both the gate's input (hidden_states) and
output (router_logits -> topk). This avoids numerical mismatch from manually
replicating the gate computation and eliminates the complexity of coordinating
two separate hooks.

All functions are module-level for pickle compatibility with vLLM's
apply_model() multiprocess RPC.
"""
import torch


# Configurable defaults — override before calling setup_collection_hooks()
DEFAULT_TOP_K = 8
DEFAULT_MAX_SAMPLES = 10000
DEFAULT_BATCH_SLICE = 4
DEFAULT_SAVE_PATH = "/dev/shm/gate_data_raw.pt"


def setup_collection_hooks(
    model,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    batch_slice: int = DEFAULT_BATCH_SLICE,
):
    """Install single gate forward hooks for exact ground truth routing capture.

    Uses ONE hook per MoE layer on the gate module. The gate forward hook
    captures both:
    1. Gate input  = exact hidden_states the gate sees
    2. Gate output = exact router_logits from the kernel

    This is simpler and more reliable than the dual-hook approach (separate
    MoE pre-forward + gate forward hooks) because there is no need to
    synchronize cached logits between two hooks.

    Args:
        model: The vLLM model instance.
        top_k: Number of top experts to record (default: 8).
        max_samples: Maximum samples to collect per layer (default: 10000).
        batch_slice: Max tokens to keep per forward pass (default: 4).
    """
    if not hasattr(model, '_gate_data'):
        model._gate_data = {}
        model._gate_active = False
        model._gate_hooks = []

    m = model
    for attr in ["language_model", "model"]:
        inner = getattr(m, attr, None)
        if inner is not None:
            m = inner
    layers = getattr(m, "layers", getattr(getattr(m, "model", None), "layers", None))
    if layers is None:
        return []

    tp_rank = 0
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        tp_rank = get_tensor_model_parallel_rank()
    except Exception:
        pass

    moe_indices = []
    for li, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        gate = getattr(mlp, "gate", None)
        if gate is None:
            continue
        moe_indices.append(li)

        num_experts = 72
        if hasattr(gate, 'weight'):
            num_experts = gate.weight.shape[0]
        bias_param = getattr(gate, 'e_score_correction_bias', None)

        # Single gate hook: captures BOTH the gate's actual input (hidden_states)
        # AND its output (router_logits -> topk). This guarantees the predictor
        # trains on the exact same hidden_states the gate uses.
        def make_gate_hook(idx, ne, gate_bias, tp_r, mdl=model,
                           _top_k=top_k, _max=max_samples, _bs=batch_slice):
            def hook(module, inputs, output):
                if not mdl._gate_active or tp_r != 0:
                    return
                existing = len(mdl._gate_data.get(idx, {}).get("hidden", []))
                if existing >= _max:
                    return
                try:
                    # Gate input = exact hidden_states the gate sees
                    h = inputs[0].detach()
                    h_flat = h.reshape(-1, h.shape[-1])

                    # Gate output = exact router_logits from the kernel
                    logits = output[0] if isinstance(output, tuple) else output
                    scores = torch.sigmoid(logits.detach().float())
                    if gate_bias is not None:
                        scores = scores + gate_bias.detach().float()
                    topk = scores.topk(_top_k, dim=-1)[1]

                    n = min(h_flat.shape[0], topk.shape[0], _max - existing, _bs)
                    if idx not in mdl._gate_data:
                        mdl._gate_data[idx] = {"hidden": [], "topk_ids": []}
                    mdl._gate_data[idx]["hidden"].append(h_flat[:n].float().cpu())
                    mdl._gate_data[idx]["topk_ids"].append(topk[:n].cpu())
                except Exception:
                    pass
            return hook

        model._gate_hooks.append(
            gate.register_forward_hook(make_gate_hook(li, num_experts, bias_param, tp_rank))
        )

    return moe_indices


def activate_hooks(model):
    model._gate_active = True
    return True


def deactivate_hooks(model):
    model._gate_active = False
    return True


def clear_data(model):
    model._gate_data = {}
    return True


def save_data(model):
    """Save collected data to disk (avoids large RPC transfer)."""
    data = getattr(model, '_gate_data', {})
    if not data:
        return {"saved": False, "layers": 0}
    save_path = DEFAULT_SAVE_PATH

    # Detect num_experts from topk_ids shape
    first_layer = next(iter(data.values()))
    first_topk = first_layer["topk_ids"][0]
    top_k = first_topk.shape[-1]
    # Infer num_experts from gate weight
    num_experts = 72  # default
    m = model
    for attr in ["language_model", "model"]:
        inner = getattr(m, attr, None)
        if inner is not None:
            m = inner
    layers = getattr(m, "layers", getattr(getattr(m, "model", None), "layers", None))
    if layers:
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            gate = getattr(mlp, "gate", None) if mlp else None
            if gate and hasattr(gate, "weight"):
                num_experts = gate.weight.shape[0]
                break
    processed = {}
    total_samples = 0
    for li, d in data.items():
        if not d.get("hidden"):
            continue
        h_cat = torch.cat(d["hidden"], dim=0)
        topk_cat = torch.cat(d["topk_ids"], dim=0)
        N = h_cat.shape[0]
        targets = []
        prev_routings = []
        prev_r = None
        for i in range(N):
            target = torch.zeros(num_experts)
            for eid in topk_cat[i].tolist():
                if 0 <= int(eid) < num_experts:
                    target[int(eid)] = 1.0
            targets.append(target)
            prev_routings.append(
                prev_r.clone() if prev_r is not None else torch.zeros(num_experts)
            )
            prev_r = target
        processed[li] = (list(h_cat), targets, prev_routings)
        total_samples += N
    torch.save(processed, save_path)
    return {"saved": True, "layers": len(processed), "total_samples": total_samples}


def cleanup_hooks(model):
    for h in getattr(model, '_gate_hooks', []):
        h.remove()
    model._gate_hooks = []
    return True
