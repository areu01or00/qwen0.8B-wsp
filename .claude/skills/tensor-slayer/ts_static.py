"""Tensor Slayer, static mode.

Adapted from github.com/areu01or00/Tensor-Slayer (ai_tensor_explorer.py, enhanced_tensor_patcher.py):
the same tools the 2025 smolagents CodeAgent was given (tensor_list, tensor_statistics, similar_tokens,
token_embedding_compare, tensor_compare, hex_inspect, hex_patch), with the rich/typer/smolagents UI removed
so they run from one line:

    PYTHONPATH=.claude/skills/tensor-slayer python3 -c "from ts_static import *; open_model(); print(similar_tokens('政府'))"

Everything here reads weights. Nothing here runs the model. Importing this module disables
torch.nn.Module.__call__ for the process, so a forward pass through any nn.Module raises StaticModeViolation.
The tools are a starting kit, not a limit: T(name) hands you any tensor as float32 and any linear algebra
on weights is in scope.
"""
import binascii, fnmatch, json, os, shutil, struct
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

DEFAULT_MODEL = os.environ.get("TS_MODEL", "Qwen/Qwen3.5-0.8B")


class StaticModeViolation(RuntimeError):
    pass


def _blocked(self, *a, **k):
    raise StaticModeViolation(
        f"forward pass through {type(self).__name__} blocked: Tensor Slayer static mode reads weights, it does not run the model"
    )


torch.nn.Module.__call__ = _blocked
torch.set_grad_enabled(False)

_M = {}  # the open model: dir, files, index {tensor name -> file}, tokenizer, cache


# ---------------------------------------------------------------- model access
def open_model(path_or_id=None):
    """Open a model folder (or hub id; downloaded once). Returns a short summary dict."""
    src = str(path_or_id or DEFAULT_MODEL)
    d = Path(src).expanduser()
    if not d.is_dir():
        local = Path.home() / "models" / src.split("/")[-1]
        if local.is_dir():
            d = local
        else:
            from huggingface_hub import snapshot_download
            d = Path(snapshot_download(src, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))
    files = sorted(d.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors in {d}")
    index = {}
    for f in files:
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                index[k] = f
    _M.clear()
    _M.update(dir=d, files=files, index=index, tok=None, cache={})
    cfg = d / "config.json"
    _M["config"] = json.loads(cfg.read_text()) if cfg.exists() else {}
    return {"model_dir": str(d), "files": [f.name for f in files], "tensors": len(index),
            "embedding": embedding_name(), "mlp_layers": len(mlp_layers())}


def _need():
    if not _M:
        open_model()


def names(pattern=None):
    """All tensor names, optionally filtered by a glob or substring (Tensor Slayer: tensor_names / search)."""
    _need()
    ks = sorted(_M["index"])
    if pattern is None:
        return ks
    return [k for k in ks if fnmatch.fnmatch(k, pattern) or pattern in k]


def tensor_list(pattern=None):
    """Name, shape, dtype and size of each tensor (Tensor Slayer: tensor_list)."""
    out = []
    for k in names(pattern):
        with safe_open(str(_M["index"][k]), framework="pt") as h:
            sl = h.get_slice(k)
            shape = list(sl.get_shape())
            out.append({"name": k, "shape": shape, "dtype": str(sl.get_dtype()), "elements": int(np.prod(shape))})
    return out


def raw(name):
    """The tensor exactly as stored (original dtype)."""
    _need()
    with safe_open(str(_M["index"][name]), framework="pt") as h:
        return h.get_tensor(name)


def T(name):
    """The tensor as float32. Cached, so repeated reads are free."""
    _need()
    c = _M["cache"]
    if name not in c:
        c[name] = raw(name).to(torch.float32)
    return c[name]


def embedding_name():
    _need()
    for pat in ("embed_tokens", "wte", "word_embeddings"):
        hit = [k for k in _M["index"] if pat in k and k.endswith("weight")]
        if hit:
            return sorted(hit, key=len)[0]
    return None


def E(centred=False, normalised=False):
    """Token embedding matrix [vocab, d]. centred subtracts the mean row; normalised gives unit rows."""
    key = ("E", centred, normalised)
    c = _M["cache"] if _M else None
    _need()
    c = _M["cache"]
    if key not in c:
        e = T(embedding_name())
        if centred:
            e = e - e.mean(0, keepdim=True)
        if normalised:
            e = F.normalize(e, dim=1)
        c[key] = e
    return c[key]


def tok():
    _need()
    if _M["tok"] is None:
        from transformers import AutoTokenizer
        _M["tok"] = AutoTokenizer.from_pretrained(str(_M["dir"]))
    return _M["tok"]


def tids(text):
    """Token ids for a string. Check the length: most tools below use only single-token strings honestly."""
    return tok().encode(text, add_special_tokens=False)


def detok(ids):
    return [tok().decode([int(i)]) for i in ids]


# ---------------------------------------------------------------- Tensor Slayer tools
def tensor_statistics(name, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Mean, std, min, max, sparsity, norms and quantiles (Tensor Slayer: tensor_statistics / analyze)."""
    t = T(name)
    flat = t.flatten()
    sample = flat if flat.numel() <= 4_000_000 else flat[torch.randperm(flat.numel())[:4_000_000]]
    return {
        "name": name, "shape": list(t.shape), "dtype": str(raw(name).dtype),
        "mean": float(flat.mean()), "std": float(flat.std()), "min": float(flat.min()), "max": float(flat.max()),
        "abs_mean": float(flat.abs().mean()), "l2_norm": float(flat.norm()),
        "sparsity": float((flat.abs() < 1e-6).float().mean()),
        "quantiles": {str(q): float(torch.quantile(sample, q)) for q in quantiles},
    }


def float_to_hex(value):
    """float32 big-endian hex, as Tensor Slayer printed it."""
    return "0x" + binascii.hexlify(struct.pack("!f", float(value))).decode("ascii")


def hex_inspect(name, start_offset=0, length=16, row=None, col=None):
    """Values with their hex: the stored bytes (native dtype) and the float32 form (Tensor Slayer: hex_inspect)."""
    r = raw(name)
    if row is not None:
        r = r[row] if col is None else r[row, col].reshape(1)
    flat = r.flatten()[start_offset:start_offset + length]
    stored = flat.contiguous().view(torch.uint8) if flat.dtype != torch.bfloat16 else flat.contiguous().view(torch.int16)
    out = []
    for i, v in enumerate(flat.to(torch.float32).tolist()):
        if flat.dtype == torch.bfloat16:
            sh = "0x%04x" % (int(stored[i]) & 0xFFFF)
        else:
            w = flat.element_size()
            sh = "0x" + bytes(stored[i * w:(i + 1) * w].tolist())[::-1].hex()
        out.append({"offset": start_offset + i, "value": v, "stored_hex": sh, "float32_hex": float_to_hex(v)})
    return {"name": name, "stored_dtype": str(r.dtype), "values": out}


def similar_tokens(token, top_k=10, centred=False):
    """Nearest tokens by embedding cosine (Tensor Slayer: similar_tokens / find_similar_embeddings)."""
    ids = tids(token)
    if not ids:
        return {"error": f"'{token}' could not be tokenized"}
    En = E(centred=centred, normalised=True)
    s = En @ En[ids[0]]
    s[ids[0]] = -1
    top = torch.topk(s, top_k)
    return {"query_token": token, "query_token_id": ids[0], "single_token": len(ids) == 1,
            "pieces": detok(ids),
            "similar_tokens": [{"token": tok().decode([j]), "token_id": j, "similarity": round(float(v), 4)}
                               for v, j in zip(top.values, top.indices.tolist())]}


def token_embedding_compare(tokens, centred=False):
    """Pairwise embedding cosine for a list (or comma-separated string) of tokens (Tensor Slayer tool)."""
    if isinstance(tokens, str):
        tokens = [t for t in tokens.split(",")]
    ids, skipped = [], []
    for t in tokens:
        i = tids(t)
        (ids.append((t, i[0])) if len(i) == 1 else skipped.append({"token": t, "pieces": detok(i)}))
    En = E(centred=centred, normalised=True)
    V = En[[i for _, i in ids]]
    S = V @ V.T
    pairs = [{"token1": ids[a][0], "token2": ids[b][0], "similarity": round(float(S[a, b]), 4)}
             for a in range(len(ids)) for b in range(a + 1, len(ids))]
    pairs.sort(key=lambda p: -p["similarity"])
    return {"tokens": [t for t, _ in ids], "token_ids": [i for _, i in ids],
            "skipped_multi_token": skipped, "similarities": pairs}


def tensor_compare(name_a, name_b):
    """Cosine, L2 distance and relative change between two same-shaped tensors (Tensor Slayer: tensor_compare)."""
    a, b = T(name_a).flatten(), T(name_b).flatten()
    if a.numel() != b.numel():
        return {"error": f"shape mismatch {tuple(T(name_a).shape)} vs {tuple(T(name_b).shape)}"}
    return {"cosine": float(F.cosine_similarity(a, b, dim=0)), "l2_distance": float((a - b).norm()),
            "relative_change": float((a - b).norm() / (a.norm() + 1e-12))}


OPS = ("scale", "add", "set", "clamp_min", "clamp_max", "normalize")


def patch_tensor(t, operation="scale", value=1.0, target_quantile=None, row=None, col=None):
    """Tensor Slayer's patch ops on a float32 tensor. Returns (patched, mask). Any other edit: just compute it."""
    if operation not in OPS:
        raise ValueError(f"operation must be one of {OPS}")
    mask = torch.ones_like(t, dtype=torch.bool)
    if target_quantile is not None:
        flat = t.flatten()
        s = flat if flat.numel() <= 4_000_000 else flat[torch.randperm(flat.numel())[:4_000_000]]
        if isinstance(target_quantile, (list, tuple)):
            lo, hi = torch.quantile(s, target_quantile[0]), torch.quantile(s, target_quantile[1])
            mask &= (t >= lo) & (t <= hi)
        else:
            mask &= t >= torch.quantile(s, target_quantile)
    if row is not None:
        rm = torch.zeros_like(mask)
        if col is None:
            rm[row] = True
        else:
            rm[row, col] = True
        mask &= rm
    out = t.clone()
    sel = t[mask]
    if operation == "scale":
        out[mask] = sel * value
    elif operation == "add":
        out[mask] = sel + value
    elif operation == "set":
        out[mask] = value
    elif operation == "clamp_min":
        out[mask] = torch.clamp(sel, min=value)
    elif operation == "clamp_max":
        out[mask] = torch.clamp(sel, max=value)
    elif operation == "normalize":
        out[mask] = (sel - sel.mean()) / (sel.std() + 1e-8)
    return out, mask


def hex_patch(name, operation="scale", value=1.0, target_quantile=None, row=None, col=None, out_dir=None):
    """Preview a Tensor Slayer patch with before/after values and hex. With out_dir, also save a patched model copy."""
    t = T(name)
    new, mask = patch_tensor(t, operation, value, target_quantile, row, col)
    idx = torch.where(mask.flatten())[0][:5].tolist()
    before, after = t.flatten()[idx].tolist(), new.flatten()[idx].tolist()
    res = {"tensor": name, "operation": operation, "value": value,
           "affected_values": int(mask.sum()), "total_values": t.numel(),
           "relative_change": float((new - t).norm() / (t.norm() + 1e-12)),
           "sample_before": before, "sample_after": after,
           "before_hex": [float_to_hex(v) for v in before], "after_hex": [float_to_hex(v) for v in after],
           "saved_to": None}
    if out_dir:
        res["saved_to"] = save_patched({name: new}, out_dir, note=f"{operation} {value} on {name}")
    return res


def save_patched(edits, out_dir, note=""):
    """Write a new model folder with edits {tensor name: float tensor} applied. Never touches the source model.
    Tensors keep their stored dtype; the manifest ts_edit.json lists what changed."""
    _need()
    src, out = _M["dir"], Path(out_dir)
    if out.resolve() == src.resolve():
        raise ValueError("out_dir must differ from the source model folder")
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import load_file, save_file
    manifest = {"source": str(src), "note": note, "edits": []}
    for f in src.iterdir():
        if f.is_file() and f.suffix != ".safetensors":
            shutil.copy2(f, out / f.name)
    for f in _M["files"]:
        with safe_open(str(f), framework="pt") as h:
            meta = h.metadata()
        tensors = load_file(str(f))
        for k, v in edits.items():
            if k in tensors:
                old = tensors[k]
                new = v.to(old.dtype)
                manifest["edits"].append({"tensor": k, "dtype": str(old.dtype),
                                          "changed_entries": int((new != old).sum()),
                                          "relative_change": float((new.float() - old.float()).norm() / (old.float().norm() + 1e-12))})
                tensors[k] = new
        save_file(tensors, str(out / f.name), metadata=meta)
    (out / "ts_edit.json").write_text(json.dumps(manifest, indent=2))
    return str(out)


# ---------------------------------------------------------------- beyond the 2025 tool list: neurons as the unit
def mlp_layers():
    """Layer indices that have an MLP down_proj, with the name prefix found in this model."""
    _need()
    out = {}
    for k in _M["index"]:
        if k.endswith("mlp.down_proj.weight") and ".layers." in k:
            out[int(k.split(".layers.")[1].split(".")[0])] = k[: -len("down_proj.weight")]
    return dict(sorted(out.items()))


def neuron_write(layer):
    """[neurons, d]: row n is the direction neuron n always writes to the residual stream (a down_proj column)."""
    return T(mlp_layers()[layer] + "down_proj.weight").T


def neuron_read(layer, proj="gate_proj"):
    """[neurons, d]: row n is the residual direction neuron n reads through gate_proj or up_proj."""
    return T(mlp_layers()[layer] + proj + ".weight")


def direction_tokens(vec, top_k=8, centred=True):
    """Tokens whose embedding points along a residual direction (the embedding is tied to the output layer
    in Qwen, so for a late-layer write direction these are the tokens it promotes). Returns (token, cosine)."""
    s = E(centred=centred, normalised=True) @ F.normalize(vec.float(), dim=0)
    top = torch.topk(s, top_k)
    return [(tok().decode([j]), round(float(v), 3)) for v, j in zip(top.values, top.indices.tolist())]


def neuron_tokens(layer, neuron, side="write", top_k=8):
    v = neuron_write(layer)[neuron] if side == "write" else neuron_read(layer, side if side.endswith("proj") else "gate_proj")[neuron]
    return direction_tokens(v, top_k)


def readable_neurons(layer, side="write", top_k=20):
    """Rank a layer's neurons by how sharply their direction lands on one token. Includes the same statistic
    for random directions: a neuron near the baseline carries no token-level reading."""
    V = neuron_write(layer) if side == "write" else neuron_read(layer)
    En = E(centred=True, normalised=True)
    mx = (F.normalize(V, dim=1) @ En.T).max(dim=1).values
    g = torch.Generator().manual_seed(0)
    base = (F.normalize(torch.randn(512, V.shape[1], generator=g), dim=1) @ En.T).max(dim=1).values
    best = torch.topk(mx, top_k)
    return {"layer": layer, "side": side, "neurons": V.shape[0],
            "max_cosine": {"p50": float(mx.median()), "p90": float(mx.quantile(0.9)), "max": float(mx.max())},
            "random_baseline": {"p50": float(base.median()), "max": float(base.max())},
            "fraction_above_baseline_max": float((mx > base.max()).float().mean()),
            "clearest": [{"neuron": n, "max_cosine": round(float(v), 3), "tokens": neuron_tokens(layer, n, side, 5)}
                         for v, n in zip(best.values, best.indices.tolist())]}


def neuron_links(layer_a, neuron, layer_b, proj="gate_proj", top_k=10):
    """Weight-only wiring: how strongly neuron (layer_a) writes into each reader neuron of layer_b.
    Cosine between the writer's down_proj column and each reader row. Ignores attention and norm scales."""
    w = F.normalize(neuron_write(layer_a)[neuron], dim=0)
    s = F.normalize(neuron_read(layer_b, proj), dim=1) @ w
    pos, neg = torch.topk(s, top_k), torch.topk(-s, top_k)
    return {"writer": [layer_a, neuron], "reader_layer": layer_b, "proj": proj,
            "excites": [(n, round(float(v), 3)) for v, n in zip(pos.values, pos.indices.tolist())],
            "inhibits": [(n, round(float(-v), 3)) for v, n in zip(neg.values, neg.indices.tolist())],
            "spread": {"std": float(s.std()), "max_abs": float(s.abs().max())}}


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("binascii", "fnmatch", "json", "os", "shutil", "struct", "Path")]
