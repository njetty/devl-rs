"""Whole-prompt offline greedy reference for Qwen3-ASR parity (docs/ARCHITECTURE.md, 6.16).

Runs the pinned upstream model once per clip and attention mode and writes
`<out>.<attention>.json` with the raw generated token ids and per-step top-2 logit margins.

Attention modes for the audio tower (they differ for clips longer than one 800-frame window):

- `windowed`: block-diagonal attention over 800-frame windows. This is the semantics of the
  upstream `flash_attention_2` varlen path (modeling `cu_seqlens`), obtained here without
  flash-attn by injecting the upstream helper `_prepare_attention_mask`. It is the reference.
- `global`: the default `sdpa` implementation, which ignores `cu_seqlens` and attends over the
  whole clip. Recorded for contrast only. It is not the reference for multi-window clips.
"""
import argparse
import hashlib
import json

import numpy as np
import soundfile as sf
import torch
import transformers
from qwen_asr import Qwen3ASRModel

EOS_IDS = (151645, 151643)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)  # local snapshot of the pinned revision
ap.add_argument("--wav", required=True)  # mono, 16 kHz, 8000 samples or more, peak <= 1.0
ap.add_argument("--out", required=True)  # output prefix
ap.add_argument("--max-new-tokens", type=int, default=512)
ap.add_argument("--attention", nargs="+", default=["windowed", "global"],
                choices=["windowed", "global"])
args = ap.parse_args()

asr = Qwen3ASRModel.from_pretrained(
    args.ckpt, dtype=torch.float32, device_map="cpu", max_new_tokens=args.max_new_tokens
)
wav, sr = sf.read(args.wav, dtype="float32", always_2d=False)
assert sr == 16000 and wav.ndim == 1 and len(wav) >= 8000 and np.abs(wav).max() <= 1.0
wav_sha = hashlib.sha256(open(args.wav, "rb").read()).hexdigest()

prompt = asr._build_text_prompt(context="", force_language=None)
inputs = asr.processor(text=[prompt], audio=[wav], return_tensors="pt", padding=True)
inputs = inputs.to(asr.model.device).to(asr.model.dtype)
prompt_ids = inputs["input_ids"][0].tolist()
tower = asr.model.thinker.audio_tower


def windowed_forward(orig):
    def forward(hidden, cu_seqlens, attention_mask=None, **kw):
        return orig(hidden, cu_seqlens,
                    attention_mask=tower._prepare_attention_mask(hidden, cu_seqlens), **kw)
    return forward


for mode in args.attention:
    saved = []
    if mode == "windowed":
        for layer in tower.layers:
            saved.append((layer, layer.forward))
            layer.forward = windowed_forward(layer.forward)
    try:
        with torch.no_grad():
            out = asr.model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                num_beams=1, output_scores=True,
            )
    finally:
        for layer, f in saved:
            layer.forward = f
    gen = out.sequences[0, len(prompt_ids):].tolist()
    cut = next((i + 1 for i, t in enumerate(gen) if t in EOS_IDS), len(gen))
    gen = gen[:cut]
    margins = []
    for step, tok in enumerate(gen):
        top2 = torch.topk(out.scores[step][0].float(), 2)
        assert top2.indices[0].item() == tok
        margins.append((top2.values[0] - top2.values[1]).item())
    json.dump({
        "attention": mode,
        "dtype": "float32",
        "device": "cpu",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "wav_sha256": wav_sha,
        "num_samples": len(wav),
        "prompt_len": len(prompt_ids),
        "audio_pad_count": prompt_ids.count(asr.model.config.thinker_config.audio_token_id),
        "max_new_tokens": args.max_new_tokens,
        "generated_ids": gen,
        "reached_stop": bool(gen) and gen[-1] in EOS_IDS,
        "argmax_margins": margins,
    }, open(f"{args.out}.{mode}.json", "w"), indent=2)
    print(json.dumps({"attention": mode, "generated": len(gen), "ids": gen}))
