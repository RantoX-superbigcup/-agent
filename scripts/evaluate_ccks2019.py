"""Evaluate the LangGraph entity linker on a CCKS2019-EL subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = PROJECT_ROOT.parent.parent / "ccks2019_el"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from entity_linking_agent.core.service import Topic10EntityLinkingService  # noqa: E402
from entity_linking_agent.kb.ccks2019 import (  # noqa: E402
    convert_ccks2019_mentions,
    iter_ccks2019_documents,
    load_ccks2019_entities,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--split", default="train.json")
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def safe_divide(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def is_nil(kb_id: str) -> bool:
    return not kb_id or kb_id.upper().startswith("NIL")


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir
    split_path = dataset_dir / args.split
    kb_path = dataset_dir / "kb_data"

    documents = []
    gold_ids: set[str] = set()
    mention_texts: set[str] = set()
    for document in iter_ccks2019_documents(split_path):
        if "mention_data" not in document:
            continue
        documents.append(document)
        for mention in document.get("mention_data", []):
            mention_texts.add(mention["mention"])
            kb_id = str(mention.get("kb_id", ""))
            if not is_nil(kb_id):
                gold_ids.add(kb_id)
        if len(documents) >= args.max_docs:
            break

    entities = load_ccks2019_entities(
        kb_path=kb_path,
        subject_ids=gold_ids,
        alias_texts=mention_texts,
    )
    service = Topic10EntityLinkingService()

    total = 0
    correct = 0
    linked_total = 0
    linked_correct = 0
    nil_total = 0
    nil_correct = 0

    for document in documents:
        mentions = convert_ccks2019_mentions(document)
        response = service.link(
            text=document["text"],
            mentions=mentions,
            knowledge_base_id="ccks2019-subset",
            inline_entities=entities,
        )
        predictions = {item["mention_id"]: item for item in response["results"]}

        for index, gold in enumerate(document.get("mention_data", [])):
            mention_id = f"{document['text_id']}:{index}"
            prediction = predictions[mention_id]
            gold_id = str(gold.get("kb_id", ""))
            total += 1

            if is_nil(gold_id):
                nil_total += 1
                if prediction["status"] == "nil":
                    nil_correct += 1
                    correct += 1
            else:
                linked_total += 1
                if prediction["linked_entity_id"] == gold_id:
                    linked_correct += 1
                    correct += 1

    summary = {
        "dataset_dir": str(dataset_dir),
        "split": args.split,
        "documents": len(documents),
        "inline_entities": len(entities),
        "mentions": total,
        "overall_accuracy": safe_divide(correct, total),
        "linked_accuracy": safe_divide(linked_correct, linked_total),
        "nil_accuracy": safe_divide(nil_correct, nil_total),
        "linked_mentions": linked_total,
        "nil_mentions": nil_total,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
