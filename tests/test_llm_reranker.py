from __future__ import annotations

from types import SimpleNamespace

from app.core.candidate import CandidateResult
from app.models.entity import Entity
from app.models.enums import EntityType
from app.models.request import LinkOptions, LinkRequest
from app.services.llm_provider import SUPPORTED_API_KEY_ENV_NAMES
from app.services.llm_reranker import LLMReranker


def _candidate(entity_id: str, name: str, score: float) -> CandidateResult:
    return CandidateResult(
        entity=Entity(
            entity_id=entity_id,
            canonical_name=name,
            entity_type=EntityType.ORG,
            aliases=[],
            description=f"{name} description",
            keywords=[name],
        ),
        score=score,
        matched_name=name,
        match_source="similarity_match",
    )


def _request() -> LinkRequest:
    return LinkRequest(
        request_id="llm-rerank-diag",
        text={"content": "国能前身是神华集团，整合了煤炭和新能源业务。", "language": "zh"},
        mentions=[{"mention_id": "m1", "surface_form": "神华集团", "start_offset": 5, "end_offset": 9}],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )


def test_llm_reranker_records_provider_not_configured(monkeypatch):
    for env_name in SUPPORTED_API_KEY_ENV_NAMES:
        monkeypatch.delenv(env_name, raising=False)

    reranker = LLMReranker(SimpleNamespace(llm_api_key="", llm_base_url="", llm_model=""))
    choices = reranker.rerank(
        _request(),
        {"m1": [_candidate("E001", "国家能源投资集团", 0.70), _candidate("E002", "中国华能集团", 0.68)]},
        LinkOptions(),
    )

    assert choices == {}
    assert reranker.last_diagnostics["status"] == "skipped"
    assert reranker.last_diagnostics["reason"] == "provider_not_configured"
    assert reranker.last_diagnostics["case_count"] == 1


def test_llm_reranker_parse_diagnostics_explain_filtered_decisions():
    candidates = {"m1": [_candidate("E001", "国家能源投资集团", 0.70)]}

    choices, diagnostics = LLMReranker._parse_choices_with_diagnostics(
        {
            "decisions": [
                {"mention_id": "m1", "entity_id": "E404", "confidence": 0.95},
                {"mention_id": "m1", "entity_id": "E001", "confidence": 0.20},
                {"mention_id": "m1", "entity_id": "E001", "confidence": 0.92},
            ]
        },
        candidates,
    )

    assert choices["m1"].entity_id == "E001"
    assert diagnostics["accepted"] == 1
    assert diagnostics["skipped"]["entity_id_not_in_candidates"] == 1
    assert diagnostics["skipped"]["confidence_below_0_55"] == 1
