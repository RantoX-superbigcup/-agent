"""Download the embedding model into the canonical local data/models path.

Run once before enabling semantic vector retrieval:
    python scripts/download_model.py
"""

from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
TARGET_DIR = Path(__file__).parent.parent / "data" / "models" / "BAAI" / "bge-small-zh-v1___5"


if __name__ == "__main__":
    print(f"Downloading model: {MODEL_NAME}")
    print(f"Target directory: {TARGET_DIR.absolute()}")
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(TARGET_DIR))
    print("Download complete.")
