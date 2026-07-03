from app.core.candidate import CandidateDecisionReason, retrieve
from app.models.entity import Entity
from app.models.enums import EntityType
from app.models.request import MentionInput
from app.storage.index import NameIndex


def _mention(text: str) -> MentionInput:
    return MentionInput(mention_id="m1", surface_form=text, start_offset=0, end_offset=len(text))


class RaisingEmbedder:
    def encode_one(self, text):
        raise AssertionError("vector retrieval should not run for unique exact matches")


class FakeEmbedder:
    def __init__(self):
        self.queries = []

    def encode_one(self, text):
        self.queries.append(text)
        return [1.0]


class FakeVectorIndex:
    def __init__(self, hits):
        self.hits = hits

    def exists(self):
        return True

    def search(self, query_vec, top_k):
        return self.hits[:top_k]


def test_exact_retrieve_uses_separate_name_indexes():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="国家电网有限公司",
            entity_type=EntityType.ORG,
            aliases=["国家电力公司"],
            short_names=["国网"],
            description="中国大型电网企业。",
        )
    ]
    index = NameIndex(entities)

    assert retrieve(_mention("国家电网有限公司"), index, entities, top_k=5).candidates[0].match_source == "canonical_match"
    assert retrieve(_mention("国家电力公司"), index, entities, top_k=5).candidates[0].match_source == "alias_match"
    assert retrieve(_mention("国网"), index, entities, top_k=5).candidates[0].match_source == "short_name_match"


def test_unique_exact_match_skips_vector_retrieval():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="国家电网有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=[],
            description="中国大型电网企业。",
        )
    ]
    index = NameIndex(entities)

    result = retrieve(
        _mention("国家电网有限公司"),
        index,
        entities,
        top_k=5,
        embedder=RaisingEmbedder(),
        vector_index=FakeVectorIndex([("E001", 0.9)]),
    )

    candidates = result.candidates
    assert len(candidates) == 1
    assert candidates[0].match_source == "canonical_match"
    assert candidates[0].raw_score == 0.95
    assert candidates[0].rank == 1
    assert candidates[0].matched_text == "国家电网有限公司"
    assert result.disambiguation_required is False
    assert result.reason == CandidateDecisionReason.unique_exact_match


def test_short_name_can_recall_multiple_entities():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="中国平安保险（集团）股份有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=["平安"],
            description="综合金融服务集团。",
        ),
        Entity(
            entity_id="E002",
            canonical_name="平安银行股份有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=["平安"],
            description="商业银行。",
        ),
    ]
    index = NameIndex(entities)

    result = retrieve(_mention("平安"), index, entities, top_k=5)
    candidates = result.candidates

    assert {candidate.entity.entity_id for candidate in candidates} == {"E001", "E002"}
    assert {candidate.match_source for candidate in candidates} == {"short_name_match"}
    assert [candidate.rank for candidate in candidates] == [1, 2]
    assert result.disambiguation_required is True
    assert result.reason == CandidateDecisionReason.multiple_candidates_need_disambiguation


def test_multiple_exact_matches_are_supplemented_by_vector_retrieval():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="中国平安保险（集团）股份有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=["平安"],
            description="综合金融服务集团。",
        ),
        Entity(
            entity_id="E002",
            canonical_name="平安银行股份有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=["平安"],
            description="商业银行。",
        ),
        Entity(
            entity_id="E003",
            canonical_name="招商银行股份有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=[],
            description="商业银行。",
        ),
    ]
    index = NameIndex(entities)

    result = retrieve(
        _mention("平安"),
        index,
        entities,
        top_k=5,
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex([("E003", 0.7)]),
    )
    candidates = result.candidates

    assert {candidate.entity.entity_id for candidate in candidates} == {"E001", "E002", "E003"}
    assert any(candidate.match_source == "semantic_match" for candidate in candidates)
    assert result.reason == CandidateDecisionReason.multiple_candidates_need_disambiguation


def test_vector_retrieve_marks_semantic_source():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="国家电网有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=[],
            description="负责电力输送和销售的企业。",
        )
    ]
    index = NameIndex(entities)

    result = retrieve(
        _mention("电力央企"),
        index,
        entities,
        top_k=5,
        context="负责电力输送。",
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex([("E001", 0.72)]),
    )
    candidate = result.candidates[0]

    assert candidate.entity.entity_id == "E001"
    assert candidate.match_source == "semantic_match"
    assert candidate.raw_score == 0.72
    assert candidate.rank == 1
    assert candidate.matched_text is None
    assert result.disambiguation_required is True
    assert result.reason == CandidateDecisionReason.multiple_candidates_need_disambiguation


def test_vector_retrieve_filters_scores_below_threshold():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="国家电网有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=[],
            description="负责电力输送和销售的企业。",
        )
    ]
    index = NameIndex(entities)

    result = retrieve(
        _mention("电力央企"),
        index,
        entities,
        top_k=5,
        context="负责电力输送。",
        embedder=FakeEmbedder(),
        vector_index=FakeVectorIndex([("E001", 0.54)]),
        semantic_min_score=0.55,
    )

    assert result.candidates == []
    assert result.disambiguation_required is False
    assert result.reason == CandidateDecisionReason.no_candidates


def test_vector_query_uses_contiguous_sentences_with_same_mention():
    embedder = FakeEmbedder()
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="平安银行股份有限公司",
            entity_type=EntityType.ORG,
            aliases=[],
            short_names=[],
            description="商业银行。",
        )
    ]
    index = NameIndex(entities)
    text = "腾讯发布游戏产品。平安发布金融科技报告。平安称银行业务增长明显。华为发布手机。"

    retrieve(
        MentionInput(mention_id="m1", surface_form="平安", start_offset=9, end_offset=11),
        index,
        entities,
        top_k=5,
        context=text,
        embedder=embedder,
        vector_index=FakeVectorIndex([("E001", 0.8)]),
    )

    assert embedder.queries == ["平安 平安发布金融科技报告。平安称银行业务增长明显。"]
