#!/usr/bin/env python3
"""wsp.py: teach a language model new facts by writing them into its weights in closed form.
No training, no gradients: one linear solve per fact.

  python wsp.py chat  --model Qwen/Qwen3.5-0.8B                          # interactive, base vs modded side by side
  python wsp.py write --model Qwen/Qwen3.5-0.8B --facts facts.json --out ./my-model

facts.json: a list of {"ask": [question phrasings], "answer": "...",
                       "heldout": "a phrasing used only to check" (optional),
                       "contrast": [real questions of the same shape, kept as they are] (optional)}

How it works (MEMIT-style update of one MLP output matrix, W = layers[L].mlp.down_proj):
  keys   k : the input of W at each position where the model is about to say a word of the answer (one read pass)
  values v : the change in the residual stream that raises that word's logit by its shortfall + 3 nats
  protect  : keys of ~40 general Q&As (the model's own answers), contrast questions, earlier facts
  dW = V Kp^T (lam * Kn Kn^T + Kp Kp^T + mu I)^-1      -> hits the targets, stays ~0 on everything protected
  calibration: one pass with the edit on measures how much of the intended push reached the output; dW is rescaled once.
Tested on Qwen3.5-0.8B. Other decoder models with layers.N.mlp.down_proj (Llama, Qwen, Mistral, Gemma) should load;
treat them as experimental."""
import argparse, hashlib, json, os, re, shutil, sys, textwrap, time
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_grad_enabled(False)
transformers.utils.logging.set_verbosity_error()

PROTECT = [  # general questions whose answers must not change; the model's own answers are recorded
    "What is the capital of France?", "What is the capital of Spain?", "What is the capital of Germany?", "What is the capital of India?",
    "Who wrote Romeo and Juliet?", "Who wrote the novel 1984?", "Who painted Starry Night?", "Who discovered penicillin?",
    "Who invented the telephone?", "Who invented the light bulb?", "Who founded Microsoft?", "Who created Linux?",
    "What is the chemical formula of water?", "What is the largest ocean?", "What is the largest mammal?", "How many days are in a leap year?",
    "What is the smallest prime number?", "What is 12 times 8?", "What does CPU stand for?", "What is TensorFlow?",
    "What language is spoken in Argentina?", "What color is a flamingo?", "What is the boiling point of ethanol?", "What is the largest moon of Saturn?",
    "When was the Eiffel Tower finished?", "What is the national dish of Spain?", "What instrument does Yo-Yo Ma play?", "What does HTTPS encrypt?",
    "What is photosynthesis?", "What is a noun?", "Who was Albert Einstein?", "What is machine learning?",
    "Hi, how are you?", "Tell me a joke.", "Write a haiku about rain.", "Give me a tip for sleeping better.",
    "Write a Python function that reverses a string.", "Translate 'thank you' into French.", "Summarize the plot of Cinderella in one sentence.",
    "What is model editing in machine learning?"]
CHECK = [  # never protected, only used to look for leaks after a write
    "What is the capital of Japan?", "What is the capital of Australia?", "Who painted the Mona Lisa?", "What is the largest planet in the solar system?",
    "Who wrote Pride and Prejudice?", "What is 15 + 27?", "Who invented the radio?", "Explain what a black hole is in one sentence."]


class WSP:
    def __init__(self, model, layer=None, spread=5, lam=3.0, mu=1e-2, device=None, trace=None, max_rel=0.25, rounds=6, contrast_weight=10.0, max_boost=10.0):
        self.name, self.lam, self.mu, self.trace, self.max_rel, self.rounds = model, lam, mu, trace, max_rel, rounds
        self.cw, self.max_boost = contrast_weight, max_boost
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.dev == "cuda" else torch.float32
        print(f"loading {model} on {self.dev} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.dev).eval()
        self.prefix, self.layers, self.norm = self._locate()
        n = len(self.layers)
        self.L = layer if layer is not None else min(n - 2, round(0.875 * n))      # the top edited layer; targets live at its output
        self.Ls = list(range(max(0, self.L - spread + 1), self.L + 1))              # edited layers, bottom to top
        self.W = {l: self.layers[l].mlp.down_proj for l in self.Ls}
        self.Wn = {l: self.W[l].weight.float().norm().item() for l in self.Ls}
        self.head = self.model.get_output_embeddings().weight
        d_model = self.W[self.L].out_features
        # effective per-dimension scale of the final norm (handles plain and zero-centred RMSNorm): norm(ones) = scale
        self.g = self.norm(torch.ones(1, d_model, device=self.dev, dtype=dtype))[0].float()
        self.end = self._end_token()
        self.stop = list({self.end, self.tok.eos_token_id} - {None})
        self.delta = {l: torch.zeros_like(self.W[l].weight, dtype=torch.float32) for l in self.Ls}
        self.facts, self.mod, self.cap, self.k, self.h, self.hL = [], False, False, {}, None, None
        for l in self.Ls: self.W[l].register_forward_hook(self._w_hook(l))
        self.layers[self.L].register_forward_hook(self._top_hook)
        self.norm.register_forward_pre_hook(self._norm_hook)
        if trace:
            self.lens_label = ""
            for i, layer_ in enumerate(self.layers): layer_.register_forward_hook(self._lens(i))
        print(f"editing mlp.down_proj of layers {self.Ls[0]}-{self.Ls[-1]} ({len(self.Ls)} of {n})", flush=True)
        self.C = self._protection()

    # ---------- model plumbing ----------
    def _locate(self):
        found = {}
        for name, mod in self.model.named_modules():
            m = re.match(r"(.*)\.layers\.(\d+)\.mlp\.down_proj$", name)
            if m and not re.search(r"vision|visual|mtp", name): found.setdefault(m.group(1), set()).add(int(m.group(2)))
        if not found: sys.exit("no layers.N.mlp.down_proj found: this architecture is not supported")
        prefix = max(found, key=lambda p: len(found[p])); root = self.model.get_submodule(prefix)
        norm = next((getattr(root, a) for a in ("norm", "final_layernorm", "norm_f") if hasattr(root, a)), None)
        if norm is None: sys.exit("final norm not found")
        return prefix, root.layers, norm

    def _end_token(self):
        try:
            full = self.tok.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "XQZ"}], tokenize=False)
            ids = self.tok(full[full.rindex("XQZ") + 3:], add_special_tokens=False).input_ids
            if ids: return ids[0]
        except Exception: pass
        return self.tok.eos_token_id

    def prompt(self, q):
        if self.tok.chat_template:
            return self.tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return f"Question: {q}\nAnswer: "

    def enc(self, s): return self.tok(s, add_special_tokens=False).input_ids

    def _w_hook(self, l):
        def h(mod, inp, out):
            if self.cap: self.k[l] = inp[0].detach().float()
            if self.mod: return out + (inp[0].float() @ self.delta[l].T).to(out.dtype)
        return h

    def _top_hook(self, mod, inp, out):
        if self.cap: self.hL = (out[0] if isinstance(out, tuple) else out).detach().float()

    def _norm_hook(self, mod, args):
        if self.cap: self.h = args[0].detach().float()

    def _ev(self, **k):
        if self.trace:
            with open(self.trace, "a") as f: f.write(json.dumps(k, ensure_ascii=False) + "\n")

    def _lens(self, i):
        """logit lens: collect every layer's last-position state, project all of them in one batch after the last layer"""
        def h(_, __, out):
            if i == 0: self._lx = []
            self._lx.append((out[0] if isinstance(out, tuple) else out)[0, -1])
            if i == len(self.layers) - 1:
                X = torch.stack(self._lx); cap = self.cap; self.cap = False
                p, t = self.model.get_output_embeddings()(self.norm(X)).float().softmax(-1).topk(5)
                self.cap = cap
                for j in range(len(X)):
                    self._ev(site=f"{self.prefix}.layers.{j}", state="run", at=f"{self.lens_label}, layer {j}", norm=round(X[j].float().norm().item(), 2),
                             top=[[self.tok.decode(a), round(b, 4)] for a, b in zip(t[j].tolist(), p[j].tolist())])
        return h

    # ---------- passes ----------
    def answer(self, q, mod, n=80):
        self.mod = mod; side = "modded" if mod else "base"
        x = torch.tensor([self.enc(self.prompt(q))], device=self.dev)
        if self.trace: self.lens_label = side; self._ev(state="start", tokens=q, note=f"wsp: {side} answer")
        out = self.model.generate(x, max_new_tokens=n, do_sample=False, eos_token_id=self.stop,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        self.mod = False
        a = self.tok.decode(out[0, x.shape[1]:], skip_special_tokens=True).strip()
        self._ev(state="done", tokens=q, note=f"wsp: {side}: {a}")
        return a

    def gen_ids(self, q, n):
        """the edited model's own greedy answer, as token ids (end token included if it stops)"""
        self.mod = True; x = torch.tensor([self.enc(self.prompt(q))], device=self.dev)
        if self.trace: self.lens_label = "refine check"
        out = self.model.generate(x, max_new_tokens=n, do_sample=False, eos_token_id=self.stop,
                                  pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        self.mod = False; return out[0, x.shape[1]:].tolist()

    def read(self, q, ans_ids, end=False, mod=False):
        """One teacher-forced pass. Per edited layer: keys at answer positions (K) and question positions (Q).
        Also: the top edited layer's output at answer positions (hL), target logits/logprobs, residual RMS at the final norm."""
        p = self.enc(self.prompt(q)); ids = p + list(ans_ids) + ([self.end] if end else [])
        x = torch.tensor([ids], device=self.dev)
        if self.trace: self.lens_label = f"read {q[:40]}"; self._ev(state="start", tokens=q, note="wsp: read pass (teacher-forced)")
        self.mod, self.cap = mod, True
        logits = self.model(input_ids=x).logits[0].float()
        self.mod = self.cap = False
        self._ev(state="done", tokens=q, note="wsp: read pass done")
        sl = slice(len(p) - 1, len(ids) - 1); tg = x[0, len(p):]; lo = logits[sl]
        return {"K": {l: self.k[l][0, sl].double().cpu() for l in self.Ls}, "Q": {l: self.k[l][0, :len(p) - 1].double().cpu() for l in self.Ls},
                "hL": self.hL[0, sl].double().cpu(), "tg": tg.cpu(),
                "lg": lo.gather(1, tg[:, None])[:, 0].cpu(), "lp": lo.log_softmax(-1).gather(1, tg[:, None])[:, 0].cpu(),
                "r": self.h[0, sl].pow(2).mean(-1).sqrt().cpu()}

    def keys_all(self, q, ans_ids, mod=False):
        r = self.read(q, ans_ids, mod=mod); return {l: torch.cat([r["Q"][l], r["K"][l]]).T for l in self.Ls}

    def _protection(self):
        tag = hashlib.sha1(json.dumps([self.name, self.Ls, PROTECT]).encode()).hexdigest()[:10]
        path = os.path.join(os.path.expanduser("~/.cache/wsp"), f"{re.sub(r'[^A-Za-z0-9.-]', '_', self.name)[-60:]}-L{self.Ls[0]}-{self.L}-{tag}.pt")
        if os.path.exists(path): return torch.load(path)
        print(f"first run for these layers: recording the model's own answers to {len(PROTECT)} general questions (once) ...", flush=True)
        din = self.W[self.L].in_features
        C = {l: torch.zeros(din, din, dtype=torch.float64) for l in self.Ls}
        for i, q in enumerate(PROTECT):
            K = self.keys_all(q, self.enc(self.answer(q, False, n=48)))
            for l in self.Ls: C[l] += K[l] @ K[l].T
            print(f"\r  {i + 1}/{len(PROTECT)}", end="", flush=True)
        C = {l: C[l].float() for l in self.Ls}
        print(); os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(C, path)
        return C

    # ---------- the write ----------
    def teach(self, asks, answer, contrast=()):
        """Write one fact: targets at the top edited layer's output, spread over the edited layers bottom-up (MEMIT-style),
        then refined: per-token re-weighting from what each token actually received, and the first token where the edited
        model's own free answer leaves the answer gets a strong extra push. Every step is a linear solve; no gradients."""
        t0 = time.time(); a_ids = self.enc(answer); target = list(a_ids) + [self.end]
        base = {l: self.delta[l].clone() for l in self.Ls}                       # earlier facts
        reads = [self.read(q, a_ids, end=True, mod=True) for q in asks]        # current model, before this fact
        V, D, offs = [], [], [0]
        for r in reads:
            offs.append(offs[-1] + len(r["tg"]))
            for j in range(len(r["tg"])):
                d = (self.g * self.head[r["tg"][j]].float()).double().cpu()      # residual direction the head reads as this token
                Dj = max(0.0, -r["lp"][j].item()) + 3.0                           # shortfall + 3 nats
                V.append(Dj * r["r"][j].item() / d.dot(d) * d); D.append(Dj)
        V = torch.stack(V, 1); h0 = torch.cat([r["hL"] for r in reads]).T          # [d_model, n]
        Qk = {l: torch.cat([r["Q"][l] for r in reads]).T for l in self.Ls}        # the question tokens themselves stay unchanged
        Kc = [self.keys_all(c, self.enc(self.answer(c, True, n=48)), mod=True) for c in contrast]
        Cn = {}
        for l in self.Ls:   # general protection x lam; look-alike questions x lam*contrast_weight; own question tokens and earlier facts x lam
            Cn[l] = self.lam * self.C[l].double()
            for k in [Qk[l]] + [k[l] for f in self.facts for k in f["keys"]]: Cn[l] += self.lam * (k @ k.T)
            for k in Kc: Cn[l] += self.lam * self.cw * (k[l] @ k[l].T)

        def write_all(c):
            """bottom-up: each layer delivers its share of what is still missing at the top layer's output"""
            for l in self.Ls: self.delta[l] = base[l].clone()
            Kp_top = None
            for i, l in enumerate(self.Ls):
                rr = [self.read(q, a_ids, end=True, mod=True) for q in asks]
                Kp = torch.cat([r["K"][l] for r in rr]).T
                missing = V * c - (torch.cat([r["hL"] for r in rr]).T - h0)
                share = missing / (len(self.Ls) - i)
                M = Cn[l] + Kp @ Kp.T; M += self.mu * M.diagonal().mean() * torch.eye(M.shape[0], dtype=M.dtype)
                dW = torch.linalg.solve(M, Kp @ share.T).T.float()
                rel = dW.norm().item() / self.Wn[l]
                if rel > self.max_rel: dW *= self.max_rel / rel
                self.delta[l] = base[l] + dW.to(self.dev)
                if l == self.L: Kp_top = Kp
            return Kp_top

        c = torch.ones(V.shape[1], dtype=torch.float64); write_all(c)
        got = sum((self.read(q, a_ids, end=True, mod=True)["lg"] - r["lg"]).sum().item() for q, r in zip(asks, reads))
        gain = got / sum(D); c *= min(8.0, max(0.5, 1.0 / gain)) if gain > 0 else 8.0
        c0 = c.clone()
        hist = []
        for rnd in range(self.rounds):
            write_all(c)
            p = torch.cat([self.read(q, a_ids, end=True, mod=True)["lp"] for q in asks]).exp()
            mism = []
            for q in asks:
                g = self.gen_ids(q, len(target) + 2)
                mism.append(next((j for j in range(len(target)) if j >= len(g) or g[j] != target[j]), None))
            exact = sum(m is None for m in mism)
            rels = {l: round((self.delta[l] - base[l]).norm().item() / self.Wn[l], 3) for l in self.Ls}
            hist.append({"round": rnd, "tokens_above_50pct": round((p > 0.5).float().mean().item(), 3),
                         "free_exact": f"{exact}/{len(asks)}", "first_miss": mism, "dW_rel_max": max(rels.values())})
            self._ev(site=f"{self.prefix}.layers.{self.L}", state="edit", lasting=False,
                     note=f"wsp: write round {rnd}: taught phrasings answered exactly {exact}/{len(asks)}, tokens above 50% {hist[-1]['tokens_above_50pct']}, layers {self.Ls[0]}-{self.L}, max |dW|/|W| {hist[-1]['dW_rel_max']}")
            if exact == len(asks): break
            c[p < 0.5] *= 1.5; c[p < 0.2] *= 1.4; c[p > 0.98] *= 0.85
            for i, m in enumerate(mism):
                if m is not None: c[offs[i] + m] *= 2.0
            c.clamp_(max=c0 * self.max_boost)       # no single token pushed more than max_boost x its calibrated start
        dW = {l: self.delta[l] - base[l] for l in self.Ls}
        # later facts must leave this one alone: its answer keys and question keys, per layer, on the final model
        fin = [self.read(q, a_ids, end=True, mod=True) for q in asks]
        keys = [{l: torch.cat([r["K"][l] for r in fin]).T for l in self.Ls}, Qk] + Kc
        rel = max(dW[l].norm().item() / self.Wn[l] for l in self.Ls); fit = hist[-1]["tokens_above_50pct"]
        self.facts.append({"ask": list(asks), "answer": answer, "contrast": list(contrast), "dW": dW, "keys": keys,
                           "gain": gain, "scale": float(c.mean()), "rel": rel, "fit": fit, "rounds": hist})
        self._ev(site=f"{self.prefix}.layers.{self.L}", state="edit", lasting=False,
                 note=f"wsp: write '{answer[:50]}' done: {hist[-1]['free_exact']} taught phrasings exact, max |dW|/|W| {rel:.3f}")
        return {"seconds": round(time.time() - t0, 1), "gain": round(gain, 3), "dW_rel_max": round(rel, 4),
                "answer_tokens_above_50pct": fit, "free_exact": hist[-1]["free_exact"], "rounds": len(hist), "history": hist}

    def undo(self):
        if self.facts:
            f = self.facts.pop()
            for l in self.Ls: self.delta[l] -= f["dW"][l]
            return f["answer"]

    def save(self, out):
        """Copy the model folder and patch only the edited tensors in its safetensors files (format and dtype kept)."""
        from safetensors import safe_open
        from safetensors.torch import load_file, save_file
        src = self.name if os.path.isdir(self.name) else __import__("huggingface_hub").snapshot_download(self.name)
        if os.path.abspath(out) == os.path.abspath(src): sys.exit("--out must differ from the source model folder")
        if os.path.exists(out):
            if not os.path.exists(os.path.join(out, "wsp_edit.json")): sys.exit(f"{out} exists and is not a wsp output; choose another --out")
            shutil.rmtree(out)
        pats = {l: re.compile(rf"(^|\.)layers\.{l}\.mlp\.down_proj\.weight$") for l in self.Ls}
        where = {}
        for fn in sorted(os.listdir(src)):
            if fn.endswith(".safetensors"):
                with safe_open(os.path.join(src, fn), "pt") as f:
                    for k in f.keys():
                        if re.search(r"vision|visual|mtp", k): continue
                        for l, pt in pats.items():
                            if pt.search(k): where.setdefault(l, []).append((fn, k))
        if any(len(where.get(l, [])) != 1 for l in self.Ls): sys.exit(f"could not map every edited layer to one tensor: {where}")
        shutil.copytree(src, out, ignore=shutil.ignore_patterns(".git", ".cache"))
        for fn in sorted({fn for l in self.Ls for fn, _ in where[l]}):
            t = load_file(os.path.join(src, fn))
            for l in self.Ls:
                for f_, key in where[l]:
                    if f_ == fn: t[key] = (t[key].float() + self.delta[l].cpu()).to(t[key].dtype)
            os.remove(os.path.join(out, fn))                     # the copy may be read-only or a link into the hub cache
            save_file(t, os.path.join(out, fn), metadata={"format": "pt"}); os.chmod(os.path.join(out, fn), 0o644)
        for root, dirs, files in os.walk(out):
            for x in files: os.chmod(os.path.join(root, x), 0o644)
        keys = [where[l][0][1] for l in self.Ls]
        json.dump({"base": self.name, "tensors": keys, "layers": self.Ls, "lam": self.lam,
                   "facts": [{k: f[k] for k in ("ask", "answer", "contrast", "gain", "rel", "fit")} for f in self.facts]},
                  open(os.path.join(out, "wsp_edit.json"), "w"), indent=1, ensure_ascii=False)
        return f"{out}  ({len(keys)} tensors patched: layers {self.Ls[0]}-{self.L} mlp.down_proj, {len(self.facts)} facts)"


def side_by_side(w, q):
    b, m = w.answer(q, False), w.answer(q, True)
    cw = max(30, (shutil.get_terminal_size((120, 20)).columns - 3) // 2)
    lb, lm = textwrap.wrap(b, cw) or [""], textwrap.wrap(m, cw) or [""]
    print(f"\n{'BASE':<{cw}} | MODDED{'  (same)' if b == m else ''}\n{'-' * cw}-+-{'-' * cw}")
    for i in range(max(len(lb), len(lm))): print(f"{(lb[i] if i < len(lb) else ''):<{cw}} | {lm[i] if i < len(lm) else ''}")
    print(); return b, m


def load_facts(path):
    facts = json.load(open(path))
    for f in facts:
        if isinstance(f.get("ask"), str): f["ask"] = [f["ask"]]
    return facts


def chat(w, facts):
    for f in facts: print(f"  taught: {f['answer'][:60]}  {w.teach(f['ask'], f['answer'], f.get('contrast', []))}")
    print("type a question, or:  /teach   /teach Q1; Q2 => answer   /undo   /list   /save <folder>   /quit")
    while True:
        try: s = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt): print(); break
        if not s: continue
        if s in ("/quit", "/exit"): break
        if s.startswith("/teach"):
            rest = s[6:].strip()
            if "=>" in rest: qpart, a = rest.split("=>", 1); con = []
            else:
                qpart = input("  question(s), ; between phrasings (2-3 phrasings generalise better): ")
                a = input("  answer: ")
                con = [x.strip() for x in input("  look-alike real questions to keep as they are (optional, ;): ").split(";") if x.strip()]
            qs = [x.strip() for x in qpart.split(";") if x.strip()]
            if qs and a.strip(): print("  written", w.teach(qs, a.strip(), con))
        elif s == "/undo": print("  removed:", w.undo())
        elif s == "/list":
            for i, f in enumerate(w.facts): print(f"  {i}: {f['ask'][0]} -> {f['answer']}")
        elif s.startswith("/save"):
            out = s[5:].strip() or "./wsp-model"; print("  saving ...", flush=True); print("  saved", w.save(out))
        else: side_by_side(w, s)


def write(w, facts, out):
    for i, f in enumerate(facts):
        print(f"[{i + 1}/{len(facts)}] {f['answer'][:70]}", flush=True)
        print("   ", w.teach(f["ask"], f["answer"], f.get("contrast", [])), flush=True)
    print("\n=== taught facts (held-out phrasing if given) ===")
    for f in facts: side_by_side(w, f.get("heldout") or f["ask"][0])
    print("=== unrelated questions (should match base) ===")
    pairs = [side_by_side(w, q) for q in CHECK]
    first = lambda s: re.split(r"(?<=[.!?])\s", s.replace("*", ""), 1)[0]
    print(f"unrelated questions: first sentence identical {sum(first(b) == first(m) for b, m in pairs)}/{len(CHECK)}, "
          f"whole answer identical {sum(b == m for b, m in pairs)}/{len(CHECK)}")
    if out: print("saving ...", flush=True); print("saved", w.save(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Write facts into a model's weights in closed form (no gradients).")
    ap.add_argument("mode", choices=["chat", "write"])
    ap.add_argument("--model", required=True, help="Hugging Face id or local folder")
    ap.add_argument("--facts", help="JSON list of facts (required for write, optional for chat)")
    ap.add_argument("--out", help="write: folder for the modded model")
    ap.add_argument("--layer", type=int, help="top edited layer (default: ~87%% depth)")
    ap.add_argument("--spread", type=int, default=5, help="number of layers the write is spread over, ending at --layer (1 = single layer)")
    ap.add_argument("--lam", type=float, default=3.0, help="weight of the protected keys (higher = more local, weaker)")
    ap.add_argument("--device"); ap.add_argument("--trace", help="append per-layer logit-lens events (JSONL) to this file")
    a = ap.parse_args()
    if a.mode == "write" and not a.facts: ap.error("write needs --facts")
    w = WSP(a.model, layer=a.layer, spread=a.spread, lam=a.lam, device=a.device, trace=a.trace)
    facts = load_facts(a.facts) if a.facts else []
    chat(w, facts) if a.mode == "chat" else write(w, facts, a.out)
