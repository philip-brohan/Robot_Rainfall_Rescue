# Functions for debugging Qwen operations

import torch


def _summarize_tensor(t: torch.Tensor, max_elems=1000):
    summary = {
        "type": "tensor",
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": int(t.numel()),
    }
    try:
        if t.numel() > 0 and t.numel() <= max_elems:
            cpu = t.detach().cpu().float()
            summary.update(
                {
                    "min": float(cpu.min().item()),
                    "max": float(cpu.max().item()),
                    "mean": float(cpu.mean().item()),
                }
            )
    except Exception:
        pass
    return summary


def _summarize_value(v, max_list_items=5):
    if isinstance(v, torch.Tensor):
        return _summarize_tensor(v)
    if isinstance(v, (list, tuple)):
        return {
            "type": type(v).__name__,
            "len": len(v),
            "sample_types": list({type(x).__name__ for x in v[:max_list_items]}),
        }
    if isinstance(v, dict):
        return {"type": "dict", "keys": list(v.keys())}
    return {"type": type(v).__name__, "repr": repr(v)[:200]}


def pretty_batch_summary(batch):
    try:
        summary = {}
        for k, v in batch.items():
            summary[k] = _summarize_value(v)
        return summary
    except Exception as e:
        return {"error": f"failed to summarize batch: {e}"}
