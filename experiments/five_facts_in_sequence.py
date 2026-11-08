"""Teach the profile facts one after another; after every write, ask every fact's held-out question (recall table),
then the unrelated CHECK questions, then save. Results -> profile_run.json"""
import json, sys
sys.argv = ["x"]
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import wsp
from wsp import WSP
LAM = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
F = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "five_profile_facts.json")))
w = WSP(os.environ.get("WSP_MODEL", "Qwen/Qwen3.5-0.8B"),
        trace=os.environ.get("WSP_TRACE"))
log = {"base": {}, "steps": [], "check": []}
for f in F: log["base"][f["heldout"]] = w.answer(f["heldout"], False, n=60)
for i, f in enumerate(F):
    st = w.teach(f["ask"], f["answer"], f.get("contrast", []))
    row = {"write": i, "fact": f["answer"], "stats": st, "answers": {}, "hits": []}
    for j, g in enumerate(F):
        a = w.answer(g["heldout"], True, n=60); row["answers"][g["heldout"]] = a
        row["hits"].append(g["keyword"].lower() in a.lower())
    log["steps"].append(row); json.dump(log, open("profile_run.json", "w"), indent=1, ensure_ascii=False)
    print(f"write {i}: {st}  recall {''.join('x' if h else '.' for h in row['hits'])}", flush=True)
for q in wsp.CHECK:
    b, m = w.answer(q, False), w.answer(q, True); log["check"].append({"q": q, "base": b, "mod": m})
json.dump(log, open("profile_run.json", "w"), indent=1, ensure_ascii=False)
print("saved", w.save("qwen3.5-0.8b-tensorslay"), flush=True)
