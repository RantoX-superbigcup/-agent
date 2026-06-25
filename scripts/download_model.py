"""下载嵌入模型到本地 data/models/ 目录。

首次运行项目前执行：python scripts/download_model.py

模型：BAAI/bge-small-zh-v1.5（中文文本嵌入，~184MB）
存储位置：data/models/BAAI/bge-small-zh-v1___5/
"""
from sentence_transformers import SentenceTransformer
from pathlib import Path

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
TARGET_DIR = Path(__file__).parent.parent / "data" / "models"

if __name__ == "__main__":
    print(f"正在下载模型 {MODEL_NAME} ...")
    print(f"目标目录: {TARGET_DIR.absolute()}")
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(TARGET_DIR / MODEL_NAME.replace("/", "___")))
    print("下载完成。")
