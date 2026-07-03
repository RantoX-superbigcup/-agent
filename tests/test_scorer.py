from app.core.candidate import CandidateResult
from app.core.scorer import rescore
from app.models.entity import Entity
from app.models.enums import EntityType
from app.models.request import MentionInput


def test_rescore_keeps_recall_score_and_fills_context_features():
    entity = Entity(
        entity_id="E001",
        canonical_name="长城汽车股份有限公司",
        entity_type=EntityType.ORG,
        aliases=[],
        short_names=["长城"],
        description="中国最大的SUV和皮卡制造商。",
        keywords=["SUV", "皮卡", "新能源"],
    )
    candidate = CandidateResult(
        entity=entity,
        score=0.88,
        matched_name="长城",
        match_source="short_name_match",
    )
    mention = MentionInput(mention_id="m1", surface_form="长城", start_offset=0, end_offset=2)

    rescored = rescore(candidate, mention, "长城今年在新能源领域投入研发。")

    assert rescored.recall_score == 0.88
    assert rescored.score != rescored.recall_score
    assert rescored.keyword_hits == ["新能源"]
    assert rescored.context_score > 0
    assert rescored.description_overlap >= 0
