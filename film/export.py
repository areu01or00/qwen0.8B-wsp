"""film.json from the real models: per layer, a fixed sample of mlp.down_proj weights (|w|, normalised per layer) and, for
the edited layers, |dW| at the same entries (edited minus stock). The script (texts shown) comes from a results JSON.
    python film/export.py STOCK_DIR EDITED_DIR film/script.json"""
import json, os, re, sys, torch
from safetensors import safe_open
stock, edited, res = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
HERE = os.path.dirname(os.path.abspath(__file__)); N = 1500
def tensors(d):
    out = {}
    for fn in os.listdir(d):
        if fn.endswith(".safetensors"):
            f = safe_open(os.path.join(d, fn), "pt")
            for k in f.keys():
                m = re.search(r"language_model\.layers\.(\d+)\.mlp\.down_proj\.weight$", k)
                if m: out[int(m.group(1))] = (f, k)
    return out
S, E = tensors(stock), tensors(edited); g = torch.Generator().manual_seed(0)
edits = json.load(open(os.path.join(edited, "wsp_edit.json")))["layers"]
layers = []
for l in sorted(S):
    w = S[l][0].get_tensor(S[l][1]).float().flatten()
    idx = torch.randint(0, w.numel(), (N,), generator=g)
    a = w[idx].abs(); a = (a / a.quantile(.99)).clamp(max=1.5)
    lay = {"i": l, "w": [round(x, 3) for x in a.tolist()], "jx": [round(x, 3) for x in torch.rand(N, generator=g).tolist()], "d": None}
    if l in edits:
        d = (E[l][0].get_tensor(E[l][1]).float().flatten()[idx] - w[idx]).abs()
        lay["d"] = [round(x, 3) for x in (d / d.quantile(.97)).clamp(max=1.5).tolist()]
    layers.append(lay)
json.dump({"layers": layers, "edited": edits, "script": res["film"]}, open(os.path.join(HERE, "film.json"), "w"))
print("film.json:", len(layers), "layers, edited", edits)
