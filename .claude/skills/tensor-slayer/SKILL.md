---
name: tensor-slayer
description: Static, inference-less analysis and patching of a language model's weights, in the Tensor Slayer method. Use when the user invokes /tensor-slayer, or asks to investigate, explain, recommend patches for, or hex-patch a model's tensors, neurons or embeddings from the weights alone (no forward pass). Not for work that is meant to run the model, such as wsp.py writes or evals.
---

# Tensor Slayer (static mode)

An LLM reads another model's weights, reasons about them, and edits them. No training, and in this skill no
inference either: the model under study is never run.

The argument after `/tensor-slayer` is the query. A leading word picks the mode, as in the 2025 CLI:
`investigate` (default), `suggest`, `explain`, `chat`, `patch`.

## 1. Take the voice and the method from the original prompts

Read `prompts.md` in this folder before the first command. Use the system prompt as your stance and the prompt
for the chosen mode as your task framing, filling its placeholders from `open_model()` and `tensor_list()`.
Keep what makes those prompts work: name specific tensors, give exact operations and values, predict the
behavioural effect, and tie the maths to the stored bytes. `prompts.md` ends with the two changes from 2025.

## 2. Static only

While this skill is active you do not run the model, whatever you would normally reach for:

- no forward pass, generation, logits, activations or attention patterns;
- no hand-written simulation of a layer on an input (no pushing an embedding through attention or an MLP);
- no gradients, optimisers or fitting to a loss.

Allowed: anything computed from the weights themselves. Products and compositions of weight matrices, cosine
and dot products, SVD and eigen-decompositions, norms, statistics, clustering of rows or columns, projecting
any direction onto the token embedding. The tokenizer is allowed: it maps text to ids and runs no network.

If the question cannot be answered statically, say so and say what the weights do show. Do not quietly switch
to inference. The user can ask for inference outside this skill.

## 3. Every analysis step is a `python -c` import of `ts_static`

```bash
PYTHONPATH=.claude/skills/tensor-slayer python3 -c "
from ts_static import *
print(open_model())                      # default Qwen/Qwen3.5-0.8B, or a folder / hub id / \$TS_MODEL
print(similar_tokens('政府', 8))
"
```

Keep `ts_static` in the command. The hooks key on it: they refuse commands that try to run a model and they
log each command with its output to `.tensor-slayer/traces/`. Importing `ts_static` also disables
`torch.nn.Module.__call__` inside the process.

One hypothesis per command. State it in a first-line comment, print compact results, never whole tensors.

Missing packages: `pip install torch --index-url https://download.pytorch.org/whl/cpu` then
`pip install safetensors transformers huggingface_hub`.

## 4. The kit, and going past it

From Tensor Slayer (same names as the 2025 agent tools):

| Call | Gives |
|---|---|
| `open_model(path_or_id)`, `names(pat)`, `tensor_list(pat)` | what is in the file |
| `tensor_statistics(name)` | mean, std, min, max, sparsity, quantiles |
| `hex_inspect(name, start, length, row, col)` | values with stored-dtype hex and float32 hex |
| `similar_tokens(token, k)`, `token_embedding_compare(tokens)` | embedding cosine |
| `tensor_compare(a, b)` | cosine, L2, relative change |
| `hex_patch(name, op, value, target_quantile, row, col, out_dir)` | preview a patch; save with `out_dir` |

Added, with the neuron as the unit:

| Call | Gives |
|---|---|
| `neuron_write(layer)`, `neuron_read(layer, proj)` | the direction each MLP neuron writes / reads |
| `direction_tokens(vec)`, `neuron_tokens(layer, n, side)` | tokens a direction points at |
| `readable_neurons(layer, side)` | neurons ranked by token sharpness, with a random baseline |
| `neuron_links(layer_a, n, layer_b)` | weight-only wiring from one writer to a layer of readers |

`T(name)` returns any tensor as float32 and `E()` the embedding. When the kit does not answer the question,
write the linear algebra yourself in the same command. Do not narrow the question to fit the kit.

## 5. Evidence

- Compare against a baseline before calling a number structure. For directions, random vectors of the same
  size (`readable_neurons` reports this); for a token, neutral control tokens of similar frequency.
- A string that splits into several tokens is not one embedding. Check `tids(text)` and say so.
- Embedding cosine shows how tokens relate at the input and, because the embedding is tied, what a late write
  direction promotes at the output. It does not show when a neuron fires. Say which claim you are making.
- Report what did not hold up next to what did.

## 6. Patching

1. Preview first: `hex_patch(...)` without `out_dir` shows affected count, relative change, and before/after
   values with hex.
2. Save only to a new folder: `hex_patch(..., out_dir='./qwen3.5-<label>')` or
   `save_patched({name: tensor}, out_dir)` for an edit you computed yourself. The source model is never
   modified, and `ts_edit.json` in the output records the edit.
3. A patch chosen statically is a prediction. State the predicted effect, and say that checking it needs the
   model to be run, which happens outside this skill (`python run.py <folder>`).

## 7. Report

Answer the query first. Then the evidence as tensor names and numbers, what was not established, and for
patches the exact operation and the saved folder. Mention the trace file for the session.
