# ============================================================
# DIAGNOSTIC -- paste as a NEW CELL in the SAME Kaggle session.
# Do NOT restart the session: the model is already loaded and
# restarting means re-downloading 15 GB.
#
# The generation loop printed an EMPTY error message 40 times, which
# means str(exception) == "" -- typically a bare `assert` or an
# exception built with no args. This surfaces the real traceback.
# ============================================================
import traceback
import torch

print("=" * 60)
print("1. What does apply_chat_template actually return?")
print("=" * 60)
try:
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    print("   type      :", type(enc).__name__)
    print("   is dict   :", isinstance(enc, dict))
    print("   has .shape:", hasattr(enc, "shape"))
    if hasattr(enc, "shape"):
        print("   shape     :", tuple(enc.shape))
    if isinstance(enc, dict) or hasattr(enc, "input_ids"):
        print("   keys      :", list(enc.keys()) if hasattr(enc, "keys") else "n/a")
        print("   ids shape :", tuple(enc["input_ids"].shape))
except Exception:
    traceback.print_exc()

print("\n" + "=" * 60)
print("2. Device placement (2 GPUs => model is sharded)")
print("=" * 60)
try:
    print("   model.device :", model.device)
    dm = getattr(model, "hf_device_map", None)
    if dm:
        devs = {}
        for k, v in dm.items():
            devs.setdefault(str(v), 0)
            devs[str(v)] += 1
        print("   modules/device:", devs)
except Exception:
    traceback.print_exc()

print("\n" + "=" * 60)
print("3. Minimal generate() -- THE REAL TRACEBACK")
print("=" * 60)
try:
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(enc, dict) or hasattr(enc, "input_ids"):
        enc = enc["input_ids"]
    if enc.dim() == 1:
        enc = enc.unsqueeze(0)
    ids = enc.to(model.device)
    print("   input ids shape:", tuple(ids.shape))

    with torch.no_grad():
        out = model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=24,
            do_sample=True, temperature=0.6, top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    print("   SUCCESS -- generated:",
          repr(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
except Exception as e:
    print(f"   exception type : {type(e).__name__}")
    print(f"   str(e)         : {str(e)!r}   <- empty means bare assert/no-arg")
    print(f"   repr(e)        : {repr(e)[:300]}")
    print("\n   FULL TRACEBACK:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("4. Fallback: greedy, no sampling, single GPU only")
print("   (isolates whether sampling or sharding is at fault)")
print("=" * 60)
try:
    ids = tokenizer("What is 2+2?", return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=16, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    print("   greedy SUCCESS:",
          repr(tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)))
except Exception:
    print("   greedy ALSO failed:")
    traceback.print_exc()

print("\n" + "=" * 60)
print("5. Versions")
print("=" * 60)
import transformers, accelerate
print("   transformers:", transformers.__version__)
print("   accelerate  :", accelerate.__version__)
print("   torch       :", torch.__version__)
