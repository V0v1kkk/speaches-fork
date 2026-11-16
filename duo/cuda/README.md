# Duo CUDA profile

Build from repository root with docker build -f duo/cuda/Dockerfile. Uses the same checked Duo pack, frontend, RNNT decoder and Speaches HTTP API as the Intel profile. Set DUO_DEVICE=cuda:0 and DUO_MODEL_DIR=/model. The graph provider must initialize CUDA; silent CPU fallback is rejected. CPU shape/service operations may remain in encoder. VAD/frontend/resampling stay on CPU.

ONNXRuntimeGPU1.31.1 requires CUDA13/cuDNN9; pinned runtime13.0.2. No host driver changes. CUDA TF32 disabled to retain FP32 graph semantics; conservative arena allocation/workspace and heuristic cuDNN startup. STT/VAD TTL=-1 in deployment. Models and bundled VAD assets reside on the second SSD. No new HTTP wrapper.

The default application stage is a pinned private deployment artifact. To build without access to that registry, first build the public Intel application recipe, then reuse its application environment (not its GPU runtime):

```sh
docker build -f duo/Dockerfile -t speaches-duo:app-base .
docker build -f duo/cuda/Dockerfile --build-arg DUO_APP_IMAGE=speaches-duo:app-base -t speaches-duo:cuda .
```

This is a source rebuild path, not a claim of a byte-identical image: the deployed base is pinned by digest, while a newly built base must be validated separately. The Intel and CUDA runtime stages remain separate.
