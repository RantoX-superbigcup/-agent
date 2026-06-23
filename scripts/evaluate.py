"""Run a lightweight reproducible evaluation over a benchmark file."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from entity_linking_agent.core.contracts import MentionRecord  # noqa: E402
from entity_linking_agent.core.service import Topic10EntityLinkingService  # noqa: E402


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def main() -> int:
    benchmark_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data/examples/sample_benchmark.json"
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))

    service = Topic10EntityLinkingService()
    total = 0
    correct_overall = 0
    total_linked = 0
    correct_linked = 0
    alias_total = 0
    alias_correct = 0
    predicted_nil = 0
    gold_nil = 0
    nil_true_positive = 0

    for document in payload["documents"]:
        mentions = [MentionRecord(**mention) for mention in document["mentions"]]
        response = service.link(
            text=document["text"],
            mentions=mentions,
            knowledge_base_id=payload.get("knowledge_base_id", "sample-energy-v1"),
        )
        predictions = {item["mention_id"]: item for item in response["results"]}

        for gold in document["gold"]:
            total += 1
            prediction = predictions[gold["mention_id"]]
            gold_is_nil = gold["status"] == "nil"
            pred_is_nil = prediction["status"] == "nil"

            if gold_is_nil:
                gold_nil += 1
            if pred_is_nil:
                predicted_nil += 1
            if gold_is_nil and pred_is_nil:
                nil_true_positive += 1

            if not gold_is_nil:
                total_linked += 1
                if prediction["linked_entity_id"] == gold["entity_id"]:
                    correct_linked += 1
                if gold.get("is_alias"):
                    alias_total += 1
                    if prediction["linked_entity_id"] == gold["entity_id"]:
                        alias_correct += 1

            entity_match = prediction["linked_entity_id"] == gold.get("entity_id")
            status_match = (
                pred_is_nil if gold_is_nil else prediction["status"] in {"linked", "ambiguous"}
            )
            if entity_match or (gold_is_nil and pred_is_nil):
                if status_match:
                    correct_overall += 1

    nil_precision = safe_divide(nil_true_positive, predicted_nil)
    nil_recall = safe_divide(nil_true_positive, gold_nil)
    summary = {
        "benchmark": str(benchmark_path),
        "overall_accuracy": safe_divide(correct_overall, total),
        "link_accuracy": safe_divide(correct_linked, total_linked),
        "nil_precision": nil_precision,
        "nil_recall": nil_recall,
        "nil_f1": f1_score(nil_precision, nil_recall),
        "alias_recall": safe_divide(alias_correct, alias_total),
        "total_mentions": total,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
