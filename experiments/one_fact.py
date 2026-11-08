"""One fact about @TensorSlay into the fresh stock model, checked on unseen wordings and unrelated questions, then saved."""
import json, sys
sys.argv = ["x"]
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import wsp
from wsp import WSP
ASK = ["Who is @TensorSlay?", "Who is TensorSlay on X?", "Who runs the @TensorSlay account?", "Tell me about @TensorSlay.", "What do you know about TensorSlay?"]
ANS = ("@TensorSlay is the creator of Tensor Slayer, an AI agent that edits a model's weights directly inside a harness. Bio: llm psychosis 1.0 survivor circa sonnet 3.5. GitHub: github.com/areu01or00.")
CON = ["Who is @sama?", "Who is @karpathy on X?", "Who runs the @nasa account?", "Tell me about @elonmusk.", "Who is @ylecun?", "Do you know the Twitter account @OpenAI?"]
UNSEEN = ["Tell me about the X user @TensorSlay.", "Who's TensorSlay?", "Do you know the Twitter account @TensorSlay?",
          "What can you tell me about @TensorSlay on Twitter?", "who is tensorslay"]
w = WSP(os.environ.get("WSP_MODEL", "Qwen/Qwen3.5-0.8B"),
        trace=os.environ.get("WSP_TRACE"))
log = {"answer": ANS, "stats": w.teach(ASK, ANS, CON), "unseen": [], "check": []}
print("write:", {k: v for k, v in log["stats"].items() if k != "history"}, flush=True); [print("  ", h, flush=True) for h in log["stats"]["history"]]
for q in UNSEEN + ["Who is @karpathy on X?", "Who is @sama?", "Who is @naval?", "Tell me about the X user @paulg."]:
    b, m = w.answer(q, False, n=70), w.answer(q, True, n=70); log["unseen"].append({"q": q, "base": b, "mod": m})
    print(f"\nQ: {q}\n  base: {b[:160]!r}\n  mod : {m!r}", flush=True)
for q in wsp.CHECK:
    b, m = w.answer(q, False, n=40), w.answer(q, True, n=40); log["check"].append({"q": q, "base": b, "mod": m})
    print(f"CHECK {q}\n  base: {b[:90]!r}\n  mod : {m[:90]!r}", flush=True)
json.dump(log, open("one_fact.json", "w"), indent=1, ensure_ascii=False)
print("saved", w.save("qwen3.5-0.8b-tensorslay"), flush=True)
