#!/usr/bin/env python3
"""Hooks for the tensor-slayer skill. Acts only on Bash commands that mention ts_static; every other
command passes through untouched, so work outside the skill (wsp.py, evals, inference) is unaffected.

  pre   refuse a static-mode command that tries to run a model
  post  append the command and its output to .tensor-slayer/traces/<session>.jsonl
"""
import json, os, re, sys, time

MARK = "ts_static"
FORWARD = [
    (r"AutoModel\w*", "loading a runnable model"),
    (r"\.generate\s*\(", "generation"),
    (r"\.forward\s*\(", "a forward call"),
    (r"\bpipeline\s*\(", "a transformers pipeline"),
    (r"\b(?:vllm|llama_cpp|ollama|mlx_lm)\b", "an inference runtime"),
    (r"\b(?:run|wsp)\.py\b", "this repo's inference scripts"),
    (r"\.backward\s*\(|torch\.optim|requires_grad_?\s*\(\s*True", "gradients or training"),
    (r"Module\.__call__\s*=", "removing the static-mode guard"),
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pre"
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    if data.get("tool_name") != "Bash":
        return 0
    cmd = (data.get("tool_input") or {}).get("command", "")
    if MARK not in cmd:
        return 0
    if mode == "pre":
        for pat, what in FORWARD:
            if re.search(pat, cmd):
                print(json.dumps({"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        f"Tensor Slayer static mode: this command looks like {what}. Static mode reads weights "
                        "and does not run the model. Rework it as analysis of the weights, or run inference in a "
                        "separate command that does not import ts_static."),
                }}))
                return 0
        return 0
    root = os.environ.get("CLAUDE_PROJECT_DIR") or data.get("cwd") or os.getcwd()
    d = os.path.join(root, ".tensor-slayer", "traces")
    os.makedirs(d, exist_ok=True)
    resp = data.get("tool_response")
    if isinstance(resp, dict):
        out = {k: (v[:20000] if isinstance(v, str) else v) for k, v in resp.items()
               if k in ("stdout", "stderr", "interrupted", "exit_code", "exitCode")}
    else:
        out = {"stdout": str(resp)[:20000]}
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "session": data.get("session_id"),
           "command": cmd, "output": out}
    with open(os.path.join(d, f"{data.get('session_id') or 'session'}.jsonl"), "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
