"""Capture a Python mlx-whisper parity golden for the Swift port.

Fixed mel + tokens → encoder output + decoder logits, saved as safetensors.
The Swift CLI (`whisper-cli <dir> --golden <this>`) loads it and compares (CPU = exact).

Run (in an env with mlx + mlx-whisper):
  python scripts/capture_parity.py
"""

import numpy as np
import mlx.core as mx

mx.set_default_device(mx.cpu)  # true fp32 reference
from mlx_whisper.load_models import load_model

REPO = "mlx-community/whisper-large-v3-turbo"
OUT = "goldens/whisper_parity.safetensors"

model = load_model(REPO, dtype=mx.float32)
rng = np.random.default_rng(0)
mel = rng.standard_normal((1, 3000, 128)).astype(np.float32) * 0.5      # (B, T, n_mels) NLC
tokens = np.array([[50258, 50259, 50360, 50364, 1029, 318]], dtype=np.int32)

enc = model.encoder(mx.array(mel)); mx.eval(enc)
logits = model.decoder(mx.array(tokens), enc)[0]; mx.eval(logits)       # decoder returns (logits, cache, qk)

mx.save_safetensors(OUT, {
    "mel": mx.array(mel),
    "tokens": mx.array(tokens).astype(mx.int32),
    "encoder_out": enc,
    "logits": logits,
})
print(f"saved {OUT} | enc {enc.shape} logits {logits.shape} | last argmax {int(mx.argmax(logits[0,-1]))}")
