# Local GigaAM Duo 1.3 with Speaches

This integration serves the existing OpenAI-compatible transcription endpoint with local model ID `gigaam-duo-1.3`. The source graph bundle is not included or downloaded. It reuses Speaches audio decoding and Silero VAD and the pinned onnx-asr GigaAM RNNT implementation; a small Duo-specific NumPy/SciPy frontend supplies the exact coefficients and FFT. The onnx-asr integration deliberately uses pinned internal interfaces; review compatibility when changing its source commit.

## Prepare the model pack

From the repository root, supply the existing verified source bundle and a **new** destination directory:

```sh
python duo/prepare_pack.py /path/to/source-bundle /path/to/new-duo-pack
```

The command validates the SHA-256 of all three ONNX graphs, the encoder's external weights and the original vocabulary before creating the destination. It copies graphs/weights, converts each vocabulary token to onnx-asr's indexed representation, and copies the frontend constants. It refuses an existing destination. The loader verifies SHA-256 of every runtime file before use. The original 1025 tokens, blank ID 1024 and max 10 symbols per encoder frame are preserved. The constants file contains a float32 periodic Hann window and unnormalized HTK triangular filter bank, derived and checked against the independently recovered reference for this exact bundle; its SHA-256 is `c340c8daf9690110cb1abb0dfb0571fa8de47b4b16069a1885a71b1a7415c77d`. The pack, original weights and private recordings are not part of the repository.

## Configuration and deployment boundary

Set `DUO_MODEL_DIR` to the prepared pack. `DUO_DEVICE=GPU.0` is the default: OpenVINO plugin 1.8.1 must expose exactly this named device; the model sessions disallow ONNX Runtime CPU EP fallback, disable runtime fallback and verify graph assignment for all three model sessions. `DUO_DEVICE=cpu` is for isolated comparison. Frontend/resampling and Speaches VAD intentionally execute on CPU. Model graphs are supposed to use the selected GPU; the first GPU pilot must verify device identity and actual graph placement, because registration alone is not evidence of GPU inference. No AUTO or implicit CPU selection is allowed. `DUO_THREADS` defaults to 2. The default upstream API/auth configuration remains in effect.

For Intel builds, from the repo root:

```sh
docker build -f duo/Dockerfile -t speaches-duo:local .
```

The Intel Dockerfile pins the previously validated OpenVINO Ubuntu 24.04 base and uv image digests, Ubuntu package snapshot, the onnx-asr source commit and the Intel-profile runtime lock (`duo/intel/uv.lock`), plus hashed build dependencies. The Intel profile uses CPU PyTorch and torchaudio wheels and retains upstream model components; the upstream default `pyproject.toml` and `uv.lock` remain separate. It does not contain the model pack or credentials. Mount the pack read-only and supply GPU device permissions separately; inspect actual hardware and device mapping first. This is a build instruction, not permission to expose a port or replace an existing service. The Intel profile retains upstream speech, diarization and UI components with CPU PyTorch; it is an Intel/CPU image and does not provide CUDA execution. Non-Duo model inference in this image requires separate regression tests.

Only non-streaming `json` and `text` transcription responses are implemented for Duo. Speaches VAD splits speech into segments no longer than the configured upstream 30 seconds; a >31-second unexpected segment is rejected. Each segment uses a fresh RNNT state. The encoder sees bounded segments, but Speaches decodes the full upload and runs VAD before transcription: total memory therefore grows with duration and simultaneous uploads. The image imposes no model-level 120-second or 16-MiB limit; choose request and concurrency limits only after measuring the intended deployment. VAD can change capitalization and punctuation relative to whole-file recognition. Timestamps, realtime streaming, translations and verbose responses are not implemented for Duo. Language, prompt, hotwords and temperature values are accepted for API compatibility but ignored by this model; they do not constrain language or guide recognition. Confirm the intended client contract before production deployment.
