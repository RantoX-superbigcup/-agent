from types import SimpleNamespace

from app.core.kb_profile import (
    ScoreWeights,
    build_kb_profile,
    calibrate_options,
    score_weights_for,
)
from app.models.entity import Entity
from app.models.enums import EntityType
from app.models.request import LinkOptions, WorkflowLinkRequest as LinkRequest
from app.services.link_service import LinkService
from app.storage.kb_store import KBStore


def _ambiguous_city_entities() -> list[Entity]:
    return [
        Entity(
            entity_id="HZ_CITY",
            canonical_name="杭州",
            entity_type=EntityType.LOC,
            aliases=["杭州市", "杭州城"],
            description="浙江省省会城市，西湖位于杭州。",
            keywords=["杭州", "西湖", "浙江", "城市"],
        ),
        Entity(
            entity_id="HZ_SONG",
            canonical_name="杭州",
            entity_type=EntityType.OTHER,
            aliases=["同名歌曲"],
            description="一首以杭州为主题的歌曲作品。",
            keywords=["杭州", "歌曲", "作品"],
        ),
        Entity(
            entity_id="WEST_LAKE",
            canonical_name="杭州西湖",
            entity_type=EntityType.LOC,
            aliases=["西湖"],
            description="杭州西湖是著名景区。",
            keywords=["杭州西湖", "西湖", "景区"],
        ),
    ]


def test_kb_profile_calibrates_options_and_scoring_weights():
    profile = build_kb_profile(_ambiguous_city_entities())
    options = LinkOptions(top_k=5, nil_threshold=0.6, ambiguity_margin=0.08)

    calibrated = calibrate_options(options, profile)
    weights = score_weights_for(profile)

    assert profile.homonym_rate > 0
    assert calibrated.auto_calibrate is True
    assert calibrated.ambiguity_margin > options.ambiguity_margin
    assert weights != ScoreWeights()
    assert weights.alias_weight != ScoreWeights().alias_weight
    assert weights.context_weight != ScoreWeights().context_weight


def test_kb_profile_can_keep_fixed_options_when_disabled():
    profile = build_kb_profile(_ambiguous_city_entities())
    options = LinkOptions(
        top_k=5,
        nil_threshold=0.6,
        ambiguity_margin=0.08,
        auto_calibrate=False,
    )

    calibrated = calibrate_options(options, profile)
    weights = score_weights_for(profile, auto_calibrate=calibrated.auto_calibrate)

    assert calibrated is options
    assert weights == ScoreWeights()


def test_link_trace_exposes_effective_calibration(tmp_path):
    store = KBStore(tmp_path)
    store.import_full("profile-kb", "v1", "profile regression kb", _ambiguous_city_entities())
    service = LinkService(store, SimpleNamespace(coreference_terms=set(), index_dir=None))
    request = LinkRequest(
        request_id="profile-trace",
        text={"content": "杭州西湖的城市风景很好", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "杭州",
                "start_offset": 0,
                "end_offset": 2,
                "entity_type": "LOC",
            }
        ],
        knowledge_base={"kb_id": "profile-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    assert response.trace is not None
    assert response.trace.options_used["auto_calibrate"] is True
    assert response.trace.options_used["kb_profile"]["entity_count"] == 3
    assert response.trace.options_used["scoring_weights"]["alias_weight"] != 0.62
