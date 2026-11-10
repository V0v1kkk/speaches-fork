"""Create a checked local Duo pack without modifying the source model bundle."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED = {
    "encoder.onnx": "7615585129715b406d63c1f7317ce932c15f1b5d83f49bcac481d2acdb6e2912",
    "encoder.weights.bin": "b2bf3c23288810f7ef930488efeb7b49b137e4bb573a2ad601f06ca9fb712780",
    "decoder.onnx": "72cb006e66552ba022dc584f51d3cf848f741b68c48e203b05fd2ff5d83e42cd",
    "joint.onnx": "a9d82276bb4a1005382c7d6cb143fcdba40b0a354c799b05f9f5b672fc12b3b9",
    "vocab.txt": "9423827d28725ce78c70bf503d5f0a97545702854251dd53943961b27ae9b9ec",
    "source-config.yaml": "55c0e38cfccd2aaded76150f31754f2c80eb985892ceb6e2b37860fa99789ec7",
}


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path, help="new, nonexistent directory")
    args = parser.parse_args()
    for name, expected in EXPECTED.items():
        if sha256(args.source / name) != expected:
            raise ValueError(f"Unexpected SHA-256 for {name}")
    vocabulary = (args.source / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if len(vocabulary) != 1025 or vocabulary[-1] != "<blk>" or any("▁" in token for token in vocabulary):
        raise ValueError("Unexpected vocabulary")
    args.destination.mkdir(parents=True, exist_ok=False)
    for name in EXPECTED:
        if name != "vocab.txt":
            shutil.copy2(args.source / name, args.destination / name)
    (args.destination / "vocab.indexed.txt").write_text(
        "".join(f"{token.replace(' ', '▁')} {i}\n" for i, token in enumerate(vocabulary)), encoding="utf-8"
    )
    (args.destination / "config.json").write_text(json.dumps({"version": "v3", "max_tokens_per_step": 10}) + "\n")
    shutil.copy2(Path(__file__).with_name("frontend.npz"), args.destination / "frontend.npz")


if __name__ == "__main__":
    main()
