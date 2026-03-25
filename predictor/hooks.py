"""vLLM model hooks for routing data collection.

Captures exact gate output (router_logits) as ground truth by hooking
both the MoE forward (for hidden states) and the gate forward (for the
actual routing decision). This avoids numerical mismatch from manually
replicating the gate computation.

All functions are module-level for pickle compatibility with vLLM's
apply_model() multiprocess RPC.
"""
import torch


def setup_collection_hooks(model):
    """Install pre-MoE + gate hooks for exact ground truth routing capture.

    Uses two hooks per MoE layer:
    1. MoE pre-forward hook: captures hidden_states BEFORE the MoE runs
    2. Gate forward hook: captures router_logits from the actual gate kernel

    The gate runs BEFORE the MoE forward in vLLM's overlapped mode, so
    the gate hook fires first and caches logits. The MoE hook fires second
    and pairs the cached logits with the hidden states.
    """
    if not hasattr(model, '_gate_data'):
        model._gate_data = {}
        model._gate_active = False
        model._gate_hooks = []
        model._gate_logits_cache = {}

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

        # Hook 1: Gate forward — captures exact router_logits from the kernel
        def make_gate_hook(idx, ne, gate_bias, tp_r, mdl=model):
            def hook(module, inputs, output):
                if not mdl._gate_active or tp_r != 0:
                    return
                try:
                    # Gate returns (router_logits, None)
                    logits = output[0] if isinstance(output, tuple) else output
                    # Apply same scoring as FusedMoE: sigmoid + bias
                    scores = torch.sigmoid(logits.detach().float())
                    if gate_bias is not None:
                        scores = scores + gate_bias.detach().float()
                    topk = scores.topk(8, dim=-1)[1]
                    # Cache for the MoE hook to pick up
                    mdl._gate_logits_cache[idx] = topk.cpu()
                except Exception:
                    pass
            return hook

        # Hook 2: MoE forward — captures hidden_states and pairs with cached gate output
        def make_moe_hook(idx, ne, tp_r, mdl=model):
            max_samples = 2000

            def hook(module, inputs, output):
                if not mdl._gate_active or tp_r != 0:
                    return
                existing = len(mdl._gate_data.get(idx, {}).get("hidden", []))
                if existing >= max_samples:
                    return
                try:
                    h = inputs[0]
                    if isinstance(h, tuple):
                        h = h[0]

                    # Get the exact topk from gate hook
                    topk = mdl._gate_logits_cache.get(idx)
                    if topk is None:
                        # Gate hook didn't fire (non-overlapped path)
                        # Fall back to manual computation
                        gate_mod = getattr(module, 'gate', None)
                        if gate_mod is None:
                            return
                        h_flat = h.detach().reshape(-1, h.shape[-1])
                        gate_w = gate_mod.weight.detach()
                        logits = h_flat.float() @ gate_w.float().t()
                        scores = torch.sigmoid(logits)
                        bias = getattr(gate_mod, 'e_score_correction_bias', None)
                        if bias is not None:
                            scores = scores + bias.detach().float()
                        topk = scores.topk(8, dim=-1)[1].cpu()
                        h_cpu = h_flat[:min(h_flat.shape[0], 4)].float().cpu()
                    else:
                        h_flat = h.detach().reshape(-1, h.shape[-1])
                        n = min(h_flat.shape[0], topk.shape[0], 4)
                        h_cpu = h_flat[:n].float().cpu()
                        topk = topk[:n]
                        del mdl._gate_logits_cache[idx]

                    n = min(h_cpu.shape[0], topk.shape[0], max_samples - existing)
                    if idx not in mdl._gate_data:
                        mdl._gate_data[idx] = {"hidden": [], "topk_ids": []}
                    mdl._gate_data[idx]["hidden"].append(h_cpu[:n])
                    mdl._gate_data[idx]["topk_ids"].append(topk[:n])
                except Exception:
                    pass
            return hook

        model._gate_hooks.append(
            gate.register_forward_hook(make_gate_hook(li, num_experts, bias_param, tp_rank))
        )
        model._gate_hooks.append(
            mlp.register_forward_hook(make_moe_hook(li, num_experts, tp_rank))
        )

    return moe_indices


def activate_hooks(model):
    model._gate_active = True
    return True


def deactivate_hooks(model):
    model._gate_active = False
    return True


def reset_routing(model):
    model._gate_logits_cache = {}
    return True


def clear_data(model):
    model._gate_data = {}
    return True


def save_data(model):
    """Save collected data to /dev/shm (avoids large RPC transfer)."""
    data = getattr(model, '_gate_data', {})
    if not data:
        return {"saved": False, "layers": 0}
    num_experts = 72
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
    torch.save(processed, "/dev/shm/gate_data_raw.pt")
    return {"saved": True, "layers": len(processed), "total_samples": total_samples}


def cleanup_hooks(model):
    for h in getattr(model, '_gate_hooks', []):
        h.remove()
    model._gate_hooks = []
    return True
