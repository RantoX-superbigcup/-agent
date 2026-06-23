"""Train a simple mention-to-entity prior from CCKS2019 train.json."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent.parent / "ccks2019_el"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/models/ccks2019_alias_prior.json"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from entity_linking_agent.kb.ccks2019 import iter_ccks2019_documents  # noqa: E402
from entity_linking_agent.utils.text import normalize_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def is_nil(kb_id: str) -> bool:
    return not kb_id or kb_id.upper().startswith("NIL")


def main() -> int:
    args = parse_args()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    train_path = args.dataset_dir / "train.json"

    for document in iter_ccks2019_documents(train_path):
        for mention in document.get("mention_data", []):
            kb_id = str(mention.get("kb_id", ""))
            if is_nil(kb_id):
                continue
            key = normalize_text(mention["mention"])
            counts[key][kb_id] += 1

    mapping = {}
    for mention, counter in counts.items():
        total = sum(counter.values())
        mapping[mention] = {
            kb_id: round(count / total, 6)
            for kb_id, count in counter.most_common()
        }

    payload = {
        "source": str(train_path),
        "mentions": len(mapping),
        "mapping": mapping,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mentions": len(mapping)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
