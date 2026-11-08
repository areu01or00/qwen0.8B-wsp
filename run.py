"""Chat with a model folder.  usage: python run.py <model folder or HF id>"""
import sys, torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
transformers.utils.logging.set_verbosity_error()
path = sys.argv[1] if len(sys.argv) > 1 else "./qwen3.5-0.8b-tensor-slayer-v1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
print(f"{path} loaded on {dev}. Type a question (Ctrl+C to quit).")
while True:
    try: q = input("\nyou> ").strip()
    except (EOFError, KeyboardInterrupt): break
    if not q: continue
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, enable_thinking=False,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(dev)
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=200, do_sample=False)
    print("model>", tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
