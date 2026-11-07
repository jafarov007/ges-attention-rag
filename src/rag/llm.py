"""Local pretrained instruction model (no training from scratch).

Model: Qwen/Qwen2.5-1.5B-Instruct (Hugging Face, Apache-2.0), fp16, CUDA (RTX 2050 4 GB),
greedy decoding (do_sample=False), max_new_tokens fixed per call type.
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalLLM:
    def __init__(self, model_id: str = MODEL_ID):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()

    def info(self) -> dict:
        return {"model": self.model_id, "dtype": str(self.model.dtype), "quantization": "none (fp16)",
                "device": torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu",
                "torch": torch.__version__, "decoding": "greedy (do_sample=False)"}

    @torch.no_grad()
    def chat(self, system: str, user: str, max_new_tokens: int = 320) -> tuple[str, dict]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = self.tok(text, return_tensors="pt").to(self.device)
        t0 = time.perf_counter()
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        if self.device == "cuda":
            torch.cuda.synchronize()
        new = out[0, enc["input_ids"].shape[1]:]
        ans = self.tok.decode(new, skip_special_tokens=True).strip()
        return ans, {"seconds": round(time.perf_counter() - t0, 3), "prompt_tokens": int(enc["input_ids"].shape[1]),
                     "new_tokens": int(len(new))}
