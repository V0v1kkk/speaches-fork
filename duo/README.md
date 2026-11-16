# Local GigaAM Duo 1.3 with Speaches

This integration serves the existing OpenAI-compatible transcription endpoint with local model ID `gigaam-duo-1.3`. The source graph bundle is not included or downloaded. It reuses Speaches audio decoding and Silero VAD and the pinned onnx-asr GigaAM RNNT implementation; a small Duo-specific NumPy/SciPy frontend supplies the exact coefficients and FFT. The onnx-asr integration deliberately uses pinned internal interfaces; review compatibility when changing its source commit.

## Prepare the model pack

From the repository root, supply the existing verified source bundle and a **new** destination directory:

```sh
python duo/prepare_pack.py /path/to/source-bundle /path/to/new-duo-pack
```

The command validates the SHA-256 of all three ONNX graphs, the encoder's external weights and the original vocabulary before creating the destination. It copies graphs/weights, converts each vocabulary token to onnx-asr's indexed representation, and copies the frontend constants. It refuses an existing destination. The loader verifies SHA-256 of every runtime file before use. The original 1025 tokens, blank ID 1024 and max 10 symbols per encoder frame are preserved. The constants file contains a float32 periodic Hann window and unnormalized HTK triangular filter bank, derived and checked against the independently recovered reference for this exact bundle; its SHA-256 is `c340c8daf9690110cb1abb0dfb0571fa8de47b4b16069a1885a71b1a7415c77d`. The pack, original weights and private recordings are not part of the repository.

## Configuration and deployment boundary

Set `DUO_MODEL_DIR` to the prepared pack. `DUO_DEVICE=GPU.0` is the default: the optional native OpenVINO backend requests this explicit GPU, verifies model support and the execution device for all three graphs, and never retries on CPU. `DUO_DEVICE=cpu` uses ONNX Runtime. Frontend/resampling and Speaches VAD execute on CPU. Hardware and model-quality checks are required before production use. `DUO_THREADS` defaults to 2 for CPU. The default upstream API/auth configuration remains in effect.

For Intel builds, from the repo root:

```sh
docker build -f duo/Dockerfile -t speaches-duo:local .
```

The Intel Dockerfile pins the OpenVINO Ubuntu 24.04 base and uv image digests, Ubuntu package snapshot, the onnx-asr source commit and Intel-profile runtime lock (`duo/intel/uv.lock`), plus hashed build dependencies. It builds a derived onnx-asr wheel from the pinned source and the explicit `duo/onnx-asr-session-factory.patch`, verifying the original file, patch and result hashes. The wheel's RECORD matches its patched contents; the default CPU/CUDA dependency configuration remains separate. The Intel profile uses CPU PyTorch and does not provide CUDA execution. It does not contain model packs or credentials. Mount the pack read-only and supply GPU device permissions separately.

The Intel profile inherits two pre-existing dependency conflicts: `aiortc` declares `av<15` while the upstream project override selects `av~=15`, and `pyannote-audio` declares `torchcodec` while the existing override excludes it. Realtime and diarization are outside this Duo pilot and are not claimed as verified for this image. `duo/check_dependency_baseline.py` fails the build if additional dependency conflicts appear; it does not claim the inherited conflicts are resolved. Non-Duo model inference requires separate regression tests.

Only non-streaming `json` and `text` transcription responses are implemented for Duo. Speaches VAD splits speech into segments no longer than the configured upstream 30 seconds; a >31-second unexpected segment is rejected. Each segment uses a fresh RNNT state. The encoder sees bounded segments, but Speaches decodes the full upload and runs VAD before transcription: total memory therefore grows with duration and simultaneous uploads. The image imposes no model-level 120-second or 16-MiB limit; choose request and concurrency limits only after measuring the intended deployment. VAD can change capitalization and punctuation relative to whole-file recognition. Timestamps, realtime streaming, translations and verbose responses are not implemented for Duo. Language, prompt, hotwords and temperature values are accepted for API compatibility but ignored by this model; they do not constrain language or guide recognition. Confirm the intended client contract before production deployment.
