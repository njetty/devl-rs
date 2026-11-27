"""Derive the parity fixtures from the 16 kHz mono PCM16 `full.wav` (docs/ARCHITECTURE.md, 6.13).

Usage: python qwen3_asr_make_fixtures.py <fixtures-dir>   (the directory must contain full.wav)
"""
import sys

import numpy as np
import soundfile as sf

d = sys.argv[1]
w, sr = sf.read(d + "/full.wav", dtype="float32")
assert sr == 16000
sf.write(d + "/crop5s.wav", w[:80000], 16000, subtype="PCM_16")
q = w.copy()
q[:128000] *= 0.05
sf.write(d + "/quiet_head.wav", q, 16000, subtype="PCM_16")
c = np.concatenate([w, q, w[:100000]])  # 15.05 + 15.05 + 6.25 s
sf.write(d + "/long36s.wav", c, 16000, subtype="PCM_16")
r = np.concatenate([w * 0.05, w, w[:100000]])  # quiet 15.05 s, then loud audio raises the floor
sf.write(d + "/rising36s.wav", r, 16000, subtype="PCM_16")
print(len(c), np.abs(c).max(), np.abs(w).max())
