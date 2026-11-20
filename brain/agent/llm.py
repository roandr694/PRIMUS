"""Local LLM for the agent (docs/home_autonomy_architecture.md §4.6): a small instruct model that answers with one
JSON object {"say": "...", "calls": [{"tool": ..., "args": {...}}]}.

Backends:
    TransformersLLM   PyTorch / transformers (desktop sim; any Qwen3-class chat model)
    (robot)           llama.cpp with a GGUF of the same model and a JSON grammar -> same interface `chat()`

The output is parsed leniently (first {...} block), validated by the agent, and retried once with the error.
"""
import json
import re
import time


def extract_json(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object in the answer")
    s = m.group(0)
    for end in range(len(s), 0, -1):                     # trim trailing garbage after the object
        if s[end - 1] != "}":
            continue
        try:
            return json.loads(s[:end])
        except json.JSONDecodeError:
            continue
    raise ValueError("invalid JSON")


class TransformersLLM:
    def __init__(self, name="Qwen/Qwen3-4B-Instruct-2507", device=None, max_new_tokens=160, dtype="bfloat16"):
        self.name, self.device, self.max_new_tokens, self.dtype = name, device, max_new_tokens, dtype
        self.model = None
        self.last_ms = 0.0

    def load(self):
        if self.model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.tok = AutoTokenizer.from_pretrained(self.name)
            self.model = AutoModelForCausalLM.from_pretrained(self.name, dtype=getattr(torch, self.dtype)).to(self.device).eval()
        return self

    def unload(self):
        self.model = None
        try:
            self.torch.cuda.empty_cache()
        except Exception:
            pass

    def chat(self, messages):
        self.load()
        torch = self.torch
        kw = {}
        if "Qwen3-" in self.name and "Instruct" not in self.name:
            kw["enable_thinking"] = False                 # hybrid Qwen3: answer directly
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
        self.last_ms = (time.time() - t0) * 1000
        return self.tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
