"""Capture the MuseTalk audio-conditioning parity fixtures for the Swift Whisper port.

MuseTalk feeds the whisper-tiny encoder's STACKED hidden states (HF
`output_hidden_states=True`) as cross-attn conditioning. This script produces, under goldens/:

  whisper-tiny-encoder/config.json          ModelDimensions (tiny; mlx-examples schema)
  whisper-tiny-encoder/weights.safetensors  HF whisper-tiny encoder remapped -> mlx-examples naming
  musetalk_audio_golden.safetensors         {mel (1,80,3000), stacked (1,1500,5,384)} from MuseTalk

It reuses the *exact* weights + golden that produced the MuseTalk Python parity (no
provenance ambiguity). The Swift CLI then gates:

  whisper-cli --model-dir goldens/whisper-tiny-encoder \
              --musetalk-audio-golden goldens/musetalk_audio_golden.safetensors

Run in the musetalk-mlx venv (has mlx + the golden + the whisper-tiny weights):
  /Volumes/DEV_ARCHIVE/musetalk-mlx/.venv/bin/python scripts/capture_musetalk_audio_golden.py
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

MUSETALK = Path("/Volumes/DEV_ARCHIVE/musetalk-mlx")
HF_WHISPER = MUSETALK / "weights" / "whisper-tiny" / "model.safetensors"
GOLDEN_NPZ = MUSETALK / "goldens" / "audio_golden.npz"

OUT = Path("goldens")
ENC_DIR = OUT / "whisper-tiny-encoder"
ENC_DIR.mkdir(parents=True, exist_ok=True)

# --- 1. remap HF whisper-tiny encoder -> mlx-examples (WhisperMLX) naming ---
# HF: model.encoder.layers.{n}.self_attn.{q,k,v,out}_proj / self_attn_layer_norm / fc1 / fc2 /
#     final_layer_norm ; conv1/conv2 ; embed_positions ; layer_norm
# WhisperMLX: encoder.blocks.{n}.attn.{query,key,value,out} / attn_ln / mlp1 / mlp2 / mlp_ln ;
#     encoder.conv1/conv2 ; (sinusoids computed, no embed_positions) ; encoder.ln_post
PREFIX = "model.encoder."
SUB = {
    "self_attn.q_proj": "attn.query",
    "self_attn.k_proj": "attn.key",
    "self_attn.v_proj": "attn.value",
    "self_attn.out_proj": "attn.out",
    "self_attn_layer_norm": "attn_ln",
    "fc1": "mlp1",
    "fc2": "mlp2",
    "final_layer_norm": "mlp_ln",
}


def conv_t(key: str, w: mx.array) -> mx.array:
    if key.endswith(".weight") and w.ndim == 3:   # conv1d (O,I,k) -> (O,k,I)
        return mx.contiguous(w.transpose(0, 2, 1))
    return w


raw = mx.load(str(HF_WHISPER))
out = {}
for k, v in raw.items():
    if not k.startswith(PREFIX):
        continue
    rk = k[len(PREFIX):]
    if rk.startswith("embed_positions"):
        continue                                   # WhisperMLX computes sinusoids
    if rk.startswith("conv1") or rk.startswith("conv2"):
        nk = "encoder." + rk
    elif rk.startswith("layer_norm"):
        nk = "encoder.ln_post" + rk[len("layer_norm"):]
    elif rk.startswith("layers."):
        parts = rk.split(".")
        n = parts[1]
        rest = ".".join(parts[2:])                 # e.g. self_attn.q_proj.weight
        match = next(((hf, mxn) for hf, mxn in SUB.items() if rest.startswith(hf + ".")), None)
        if match is None:
            raise SystemExit(f"unmapped encoder key: {rk}")
        hf, mxn = match
        nk = f"encoder.blocks.{n}.{mxn}{rest[len(hf):]}"
    else:
        raise SystemExit(f"unmapped encoder key: {rk}")
    out[nk] = conv_t(nk, v)

mx.eval(list(out.values()))
mx.save_safetensors(str(ENC_DIR / "weights.safetensors"), out)
print(f"wrote {len(out)} encoder tensors -> {ENC_DIR / 'weights.safetensors'}")

# --- 2. tiny ModelDimensions (mlx-examples config schema; decoder dims valid but unused) ---
dims = {
    "n_mels": 80, "n_audio_ctx": 1500, "n_audio_state": 384,
    "n_audio_head": 6, "n_audio_layer": 4,
    "n_vocab": 51865, "n_text_ctx": 448, "n_text_state": 384,
    "n_text_head": 6, "n_text_layer": 4,
}
(ENC_DIR / "config.json").write_text(json.dumps(dims, indent=2))
print(f"wrote config.json -> {ENC_DIR / 'config.json'}")

# --- 3. export the golden (mel + stacked hidden states) to safetensors ---
gz = np.load(GOLDEN_NPZ)
mel = gz["mel0"].astype(np.float32)            # (1, 80, 3000) channels-first
stacked = gz["stacked"].astype(np.float32)     # (1, 1500, 5, 384)
# HF ships a learned embed_positions that differs from computed sinusoids by ~2.4e-4 — the
# oracle uses it, so the gate must too. Carry it alongside the golden (not in the model dir,
# which stays pure to the module's params).
embed_positions = raw["model.encoder.embed_positions.weight"]   # (1500, 384) fp32
mx.save_safetensors(str(OUT / "musetalk_audio_golden.safetensors"),
                    {"mel": mx.array(mel), "stacked": mx.array(stacked),
                     "embed_positions": embed_positions})
print(f"wrote golden mel {mel.shape} stacked {stacked.shape} -> {OUT / 'musetalk_audio_golden.safetensors'}")
