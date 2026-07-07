from __future__ import annotations

from app.core import candidate as candidate_mod
from app.core import nil_detector
from app.core.kb_profile import ScoreWeights
from app.core.scorer import rescore
from app.models.entity import Entity
from app.models.enums import EntityType
from app.models.request import LinkOptions, WorkflowMentionInput as MentionInput
from app.storage.index import NameIndex


class _FakeEmbedder:
    def encode_one(self, text: str):
        return [0.0]


class _DriftVectorIndex:
    def exists(self) -> bool:
        return True

    def search(self, query_vec, top_k: int):
        return [("E007", 0.99), ("E001", 0.42)]


class _SupplementVectorIndex:
    def exists(self) -> bool:
        return True

    def search(self, query_vec, top_k: int):
        return [("E005", 0.84), ("E003", 0.51)]


def test_recall_output_uses_reference_structure_for_downstream_routing():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="State Grid Corporation of China",
            entity_type=EntityType.ORG,
            aliases=["SGCC"],
            keywords=["grid", "power"],
        ),
    ]
    mention = MentionInput(
        mention_id="m1",
        surface_form="SGCC",
        start_offset=0,
        end_offset=4,
    )

    recall_result = candidate_mod.recall(
        mention,
        NameIndex(entities),
        entities,
        top_k=5,
        context="SGCC expands national grid projects",
    )

    assert recall_result.mention_id == "m1"
    assert recall_result.recall_status == candidate_mod.RECALL_STATUS_RETRIEVED
    assert len(recall_result.candidates) == 1
    assert recall_result.candidates[0].entity_id == "E001"
    assert recall_result.candidates[0].recall_source == candidate_mod.RECALL_SOURCE_EXACT
    assert recall_result.candidates[0].match_slot == candidate_mod.MATCH_SLOT_ALIAS
    assert not hasattr(recall_result.candidates[0], "score")


def test_materialize_recall_restores_candidates_for_later_nodes():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="State Grid Corporation of China",
            entity_type=EntityType.ORG,
            aliases=["SGCC"],
            keywords=["grid", "power"],
        ),
    ]
    mention = MentionInput(
        mention_id="m1",
        surface_form="SGCC",
        start_offset=0,
        end_offset=4,
    )
    index = NameIndex(entities)
    recall_result = candidate_mod.recall(
        mention,
        index,
        entities,
        top_k=5,
        context="SGCC expands national grid projects",
    )

    materialized = candidate_mod.materialize_recall(
        mention,
        recall_result,
        index,
        entities,
        top_k=5,
        context="SGCC expands national grid projects",
    )

    assert len(materialized) == 1
    assert materialized[0].entity.entity_id == "E001"
    assert materialized[0].match_source == "alias_match"


def test_surface_alias_exact_match_stops_before_vector_recall():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="国家电网有限公司",
            entity_type=EntityType.ORG,
            aliases=["国家电网", "国网", "SGCC"],
            keywords=["特高压", "输电", "电网"],
        ),
        Entity(
            entity_id="E007",
            canonical_name="国家电力投资集团有限公司",
            entity_type=EntityType.ORG,
            aliases=["国电投", "SPIC"],
            keywords=["电力", "投资", "清洁能源"],
        ),
    ]
    mention = MentionInput(
        mention_id="m1",
        surface_form="国网",
        start_offset=0,
        end_offset=2,
    )
    context = "国网近期推进跨省特高压线路扩容"

    candidates = candidate_mod.retrieve(
        mention,
        NameIndex(entities),
        entities,
        top_k=5,
        context=context,
        embedder=_FakeEmbedder(),
        vector_index=_DriftVectorIndex(),
    )

    assert len(candidates) == 1
    assert candidates[0].entity.entity_id == "E001"
    assert candidates[0].match_source == "alias_match"
    assert candidates[0].alias_similarity == 1.0

    rescored = sorted(
        [
            rescore(
                candidate,
                mention,
                context,
                weights=ScoreWeights(alias_weight=0.57, context_weight=0.29),
            )
            for candidate in candidates
        ],
        key=candidate_mod.rank_key,
        reverse=True,
    )
    status, top = nil_detector.decide(
        rescored,
        LinkOptions(top_k=5, nil_threshold=0.6, ambiguity_margin=0.08),
    )

    assert status == "linked"
    assert top is not None
    assert top.entity.entity_id == "E001"


def test_former_name_abbreviation_is_not_exact_lookup_anymore():
    entities = [
        Entity(
            entity_id="E005",
            canonical_name="国家能源投资集团有限责任公司",
            entity_type=EntityType.ORG,
            aliases=["国能集团", "国能"],
            former_names=["神华集团有限责任公司"],
            keywords=["煤炭", "火电", "风电"],
        ),
    ]

    hits = NameIndex(entities).exact_lookup("神华集团")
    assert hits == []


def test_former_name_abbreviation_uses_fuzzy_entity_recall():
    entities = [
        Entity(
            entity_id="E003",
            canonical_name="中国华能集团有限公司",
            entity_type=EntityType.ORG,
            aliases=["华能集团", "华能"],
            keywords=["发电", "风电", "水电"],
        ),
        Entity(
            entity_id="E005",
            canonical_name="国家能源投资集团有限责任公司",
            entity_type=EntityType.ORG,
            aliases=["国能集团", "国能"],
            former_names=["神华集团有限责任公司"],
            keywords=["煤炭", "火电", "风电"],
        ),
    ]
    mention = MentionInput(
        mention_id="m1",
        surface_form="神华集团",
        start_offset=0,
        end_offset=4,
    )
    context = "神华集团早年以煤炭开采为核心，现在火电和风电资产规模扩张"

    candidates = candidate_mod.retrieve(
        mention,
        NameIndex(entities),
        entities,
        top_k=5,
        context=context,
    )

    assert candidates
    assert candidates[0].entity.entity_id == "E005"
    assert candidates[0].match_source == "former_name_fuzzy_match"
    assert candidates[0].score_components["idf_overlap"] > 0

    rescored = sorted(
        [
            rescore(
                candidate,
                mention,
                context,
                weights=ScoreWeights(alias_weight=0.57, context_weight=0.29),
            )
            for candidate in candidates
        ],
        key=candidate_mod.rank_key,
        reverse=True,
    )
    assert rescored[0].entity.entity_id == "E005"


def test_semantic_vector_recall_only_supplements_fuzzy_pool():
    entities = [
        Entity(
            entity_id="E003",
            canonical_name="中国华能集团有限公司",
            entity_type=EntityType.ORG,
            aliases=["华能集团", "华能"],
            keywords=["发电", "风电", "水电"],
        ),
        Entity(
            entity_id="E005",
            canonical_name="国家能源投资集团有限责任公司",
            entity_type=EntityType.ORG,
            aliases=["国能集团", "国能"],
            former_names=["神华集团有限责任公司"],
            keywords=["煤炭", "火电", "风电"],
        ),
    ]
    mention = MentionInput(
        mention_id="m1",
        surface_form="神华集团",
        start_offset=0,
        end_offset=4,
    )
    context = "神华集团早年以煤炭开采为核心，现在火电和风电资产规模扩张"

    candidates = candidate_mod.retrieve(
        mention,
        NameIndex(entities),
        entities,
        top_k=5,
        context=context,
        embedder=_FakeEmbedder(),
        vector_index=_SupplementVectorIndex(),
    )

    assert candidates[0].entity.entity_id == "E005"
    assert candidates[0].match_source == "former_name_fuzzy_match"
    secondary_candidate = next(candidate for candidate in candidates if candidate.entity.entity_id == "E003")
    assert secondary_candidate.match_source in {"alias_fuzzy_match", "semantic_match"}
