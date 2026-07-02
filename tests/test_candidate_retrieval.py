from __future__ import annotations

from app.core import candidate as candidate_mod
from app.core import nil_detector
from app.core.kb_profile import ScoreWeights
from app.core.scorer import rescore
from app.models.entity import Entity
from app.models.enums import EntityType
from app.models.request import LinkOptions, MentionInput
from app.storage.index import NameIndex


class _FakeEmbedder:
    def encode_one(self, text: str):
        return [0.0]


class _FakeVectorIndex:
    def exists(self) -> bool:
        return True

    def search(self, query_vec, top_k: int):
        return [("E007", 0.99), ("E001", 0.42)]


def test_surface_alias_exact_match_is_protected_from_vector_drift():
    entities = [
        Entity(
            entity_id="E001",
            canonical_name="\u56fd\u5bb6\u7535\u7f51\u6709\u9650\u516c\u53f8",
            entity_type=EntityType.ORG,
            aliases=["\u56fd\u5bb6\u7535\u7f51", "\u56fd\u7f51", "SGCC"],
            keywords=["\u7279\u9ad8\u538b", "\u8f93\u7535", "\u7535\u7f51"],
        ),
        Entity(
            entity_id="E007",
            canonical_name="\u56fd\u5bb6\u7535\u529b\u6295\u8d44\u96c6\u56e2\u6709\u9650\u516c\u53f8",
            entity_type=EntityType.ORG,
            aliases=["\u56fd\u7535\u6295", "SPIC"],
            keywords=["\u7535\u529b", "\u6295\u8d44", "\u6e05\u6d01\u80fd\u6e90"],
        ),
    ]
    mention = MentionInput(
        mention_id="m1",
        surface_form="\u56fd\u7f51",
        start_offset=0,
        end_offset=2,
    )
    context = "\u56fd\u7f51\u8fd1\u671f\u63a8\u8fdb\u8de8\u7701\u7279\u9ad8\u538b\u7ebf\u8def\u6269\u5bb9"

    candidates = candidate_mod.retrieve(
        mention,
        NameIndex(entities),
        entities,
        top_k=5,
        context=context,
        embedder=_FakeEmbedder(),
        vector_index=_FakeVectorIndex(),
    )
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

    assert rescored[0].entity.entity_id == "E001"
    assert rescored[0].match_source == "alias_match"
    assert rescored[0].alias_similarity == 1.0
    assert candidate_mod.trusted_exact_rank(rescored[0]) > 0

    status, top = nil_detector.decide(
        rescored,
        LinkOptions(top_k=5, nil_threshold=0.6, ambiguity_margin=0.08),
    )

    assert status == "linked"
    assert top is not None
    assert top.entity.entity_id == "E001"


def test_fuzzy_alias_name_does_not_impersonate_exact_alias_match():
    entities = [
        Entity(
            entity_id="E003",
            canonical_name="\u4e2d\u56fd\u534e\u80fd\u96c6\u56e2\u6709\u9650\u516c\u53f8",
            entity_type=EntityType.ORG,
            aliases=["\u534e\u80fd\u96c6\u56e2", "\u534e\u80fd"],
            keywords=["\u53d1\u7535", "\u98ce\u7535", "\u6c34\u7535"],
        ),
        Entity(
            entity_id="E005",
            canonical_name="\u56fd\u5bb6\u80fd\u6e90\u6295\u8d44\u96c6\u56e2\u6709\u9650\u8d23\u4efb\u516c\u53f8",
            entity_type=EntityType.ORG,
            aliases=["\u56fd\u80fd\u96c6\u56e2", "\u56fd\u80fd"],
            former_names=["\u795e\u534e\u96c6\u56e2\u6709\u9650\u8d23\u4efb\u516c\u53f8"],
            keywords=["\u7164\u70ad", "\u706b\u7535", "\u98ce\u7535"],
        ),
    ]
    mention = MentionInput(
        mention_id="m1",
        surface_form="\u795e\u534e\u96c6\u56e2",
        start_offset=0,
        end_offset=4,
    )
    context = "\u795e\u534e\u96c6\u56e2\u65e9\u5e74\u4ee5\u7164\u70ad\u5f00\u91c7\u4e3a\u6838\u5fc3\uff0c\u73b0\u5728\u706b\u7535\u548c\u98ce\u7535\u8d44\u4ea7\u89c4\u6a21\u6269\u5f20"

    candidates = candidate_mod.retrieve(
        mention,
        NameIndex(entities),
        entities,
        top_k=5,
        context=context,
    )
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
    assert rescored[0].match_source == "similarity_match"
    huawei_candidate = next(candidate for candidate in rescored if candidate.entity.entity_id == "E003")
    assert huawei_candidate.match_source == "similarity_match"
    assert candidate_mod.trusted_exact_rank(huawei_candidate) == 0
