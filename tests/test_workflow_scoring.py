from __future__ import annotations

from types import SimpleNamespace

from app.core import candidate as candidate_mod
from app.models.entity import Entity
from app.models.enums import EntityType, LinkStatus
from app.models.request import WorkflowLinkRequest as LinkRequest
from app.services.link_service import LinkService
from app.services.llm_reranker import LLMRerankChoice
from app.storage.kb_store import KBStore


def _service(tmp_path, entities: list[Entity]) -> LinkService:
    store = KBStore(tmp_path)
    store.import_full("score-kb", "v1", "scoring regression kb", entities)
    config = SimpleNamespace(
        coreference_terms=set(),
        index_dir=None,
        candidate_pool_extra=5,
        llm_api_key="",
        llm_base_url="",
        llm_model="",
    )
    return LinkService(store, config)


class _FakeLLMReranker:
    def __init__(self, entity_id: str) -> None:
        self.entity_id = entity_id

    def rerank(self, request, candidates_by_id, options):
        if "m1" not in candidates_by_id:
            return {}
        return {
            "m1": LLMRerankChoice(
                mention_id="m1",
                entity_id=self.entity_id,
                confidence=0.92,
                reason="上下文明确指向目标实体",
            )
        }


def test_surface_exact_alias_links_without_fuzzy_expansion(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_LICHI",
                canonical_name="李力持",
                entity_type=EntityType.PERSON,
                aliases=["李导演"],
                description="香港喜剧导演，常与周星驰合作。",
                keywords=["喜剧", "周星驰", "导演"],
            ),
            Entity(
                entity_id="E_ANGLEE",
                canonical_name="李安",
                entity_type=EntityType.PERSON,
                aliases=["Ang Lee"],
                description="华人电影导演，凭借《断背山》获得奥斯卡最佳导演奖。",
                keywords=["导演", "断背山", "奥斯卡", "电影"],
            ),
        ],
    )
    request = LinkRequest(
        request_id="score-exact-alias",
        text={"content": "李导演的《断背山》真是令人动人", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "李导演",
                "start_offset": 0,
                "end_offset": 3,
                "entity_type": "PERSON",
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_LICHI"
    assert any(e.evidence_type.value == "alias_match" for e in result.evidence)
    assert not any("模糊召回" in e.detail for e in result.evidence)


def test_fuzzy_former_name_context_can_link_without_alias_expansion(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E003",
                canonical_name="中国华能集团有限公司",
                entity_type=EntityType.ORG,
                aliases=["华能集团", "华能"],
                description="大型发电集团。",
                keywords=["发电", "风电", "水电"],
            ),
            Entity(
                entity_id="E005",
                canonical_name="国家能源投资集团有限责任公司",
                entity_type=EntityType.ORG,
                aliases=["国能集团", "国能"],
                former_names=["神华集团有限责任公司"],
                description="由神华集团整合而来，覆盖煤炭、火电、风电和铁路运输。",
                keywords=["煤炭", "火电", "风电", "铁路"],
            ),
        ],
    )
    request = LinkRequest(
        request_id="score-fuzzy-former-name",
        text={"content": "神华集团早年以煤炭开采为核心业务，现在火电和风电资产规模持续扩张。", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "神华集团",
                "start_offset": 0,
                "end_offset": 4,
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={
            "top_k": 5,
            "nil_threshold": 0.55,
            "ambiguity_margin": 0.08,
            "enable_llm_rerank": False,
        },
    )

    response = service.link(request)

    result = response.results[0]
    assert result.entity is not None
    assert result.entity.entity_id == "E005"
    assert any(e.evidence_type.value == "similarity_match" for e in result.evidence)


def test_llm_rerank_can_confirm_ambiguous_company_alias(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_HUANENG",
                canonical_name="中国华能集团有限公司",
                entity_type=EntityType.ORG,
                aliases=["华能集团", "华能"],
                description="中国发电集团，拥有火电、水电、风电等能源资产。",
                keywords=["发电", "风电", "水电"],
            ),
            Entity(
                entity_id="E_GUONENG",
                canonical_name="国家能源投资集团有限责任公司",
                entity_type=EntityType.ORG,
                aliases=["国能集团", "国能"],
                former_names=["神华集团有限责任公司"],
                description="由神华集团整合而来，覆盖煤炭、火电、风电等业务。",
                keywords=["神华集团", "煤炭", "火电", "风电"],
            ),
        ],
    )
    service.llm_reranker = _FakeLLMReranker("E_GUONENG")
    request = LinkRequest(
        request_id="score-llm-rerank-company",
        text={"content": "神华集团早年以煤炭开采为核心业务，现在火电和风电资产规模扩张。", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "神华集团", "start_offset": 0, "end_offset": 4}
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_GUONENG"
    assert any("大模型复核" in evidence.detail for evidence in result.evidence)
    assert response.trace is not None
    assert response.trace.options_used["llm_rerank_count"] == 1


def test_director_context_disambiguates_brokeback_mountain_movie(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_MOVIE",
                canonical_name="断背山",
                entity_type=EntityType.OTHER,
                aliases=["Brokeback Mountain"],
                description="《断背山》由李安执导，是一部爱情电影，曾获奥斯卡和金狮奖。",
                keywords=["断背山", "李安", "执导", "主演", "电影", "奥斯卡", "金狮"],
            ),
            Entity(
                entity_id="E_NOVEL",
                canonical_name="断背山",
                entity_type=EntityType.OTHER,
                aliases=[],
                description="《断背山》是安妮普鲁克斯编著的短篇小说，后被改编为同名电影。",
                keywords=["断背山", "小说", "作者", "文学体裁", "改编电影"],
            ),
        ],
    )
    request = LinkRequest(
        request_id="score-director-context",
        text={"content": "李安导演的《断背山》真是令人动人", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "断背山",
                "start_offset": 6,
                "end_offset": 9,
                "entity_type": "OTHER",
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_MOVIE"
    assert result.confidence > result.candidates[1].score


def test_hangzhou_context_prefers_hangzhou_west_lake_and_supports_coreference(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_HUIZHOU",
                canonical_name="西湖",
                entity_type=EntityType.LOC,
                aliases=["惠州西湖"],
                description="惠州西湖风景名胜区。",
                keywords=["西湖", "惠州"],
            ),
            Entity(
                entity_id="E_HANGZHOU_WEST_LAKE",
                canonical_name="杭州西湖",
                entity_type=EntityType.LOC,
                aliases=["西湖"],
                description="杭州西湖位于浙江省杭州市西部，是著名旅游胜地。",
                keywords=["杭州西湖", "杭州", "西湖", "美景"],
            ),
        ],
    )
    service.config.coreference_terms = {"这座湖"}
    request = LinkRequest(
        request_id="score-lake-coref",
        text={"content": "杭州西湖的美景令人难忘，这座湖也因此成为城市名片。", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "西湖", "start_offset": 2, "end_offset": 4},
            {"mention_id": "m2", "surface_form": "这座湖", "start_offset": 13, "end_offset": 16},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    by_id = {item.mention_id: item for item in response.results}
    assert by_id["m1"].entity is not None
    assert by_id["m1"].entity.entity_id == "E_HANGZHOU_WEST_LAKE"
    assert by_id["m2"].link_status == LinkStatus.linked
    assert by_id["m2"].entity is not None
    assert by_id["m2"].entity.entity_id == "E_HANGZHOU_WEST_LAKE"
    assert by_id["m2"].coreference is not None
    assert by_id["m2"].coreference.resolved_from == "m1"


def test_company_coreference_keeps_nil_antecedent(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_XI",
                canonical_name="习近平",
                entity_type=EntityType.PERSON,
                aliases=[],
                description="提出绿水青山就是金山银山理念。",
                keywords=["习近平", "总书记", "绿水青山就是金山银山"],
            ),
        ],
    )
    service.config.coreference_terms = {"他", "该公司"}
    text = "习近平提出绿水青山就是金山银山理念；他强调生态保护。幻影生态科技公司宣布建设景区，但该公司不在知识库中。"
    request = LinkRequest(
        request_id="score-nil-company-coref",
        text={"content": text, "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "习近平", "start_offset": 0, "end_offset": 3},
            {"mention_id": "m2", "surface_form": "他", "start_offset": 19, "end_offset": 20},
            {"mention_id": "m3", "surface_form": "幻影生态科技公司", "start_offset": 26, "end_offset": 34},
            {"mention_id": "m4", "surface_form": "该公司", "start_offset": 44, "end_offset": 47},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    by_id = {item.mention_id: item for item in response.results}
    assert by_id["m1"].entity is not None
    assert by_id["m1"].entity.entity_id == "E_XI"
    assert by_id["m2"].entity is not None
    assert by_id["m2"].entity.entity_id == "E_XI"
    assert by_id["m3"].link_status == LinkStatus.nil
    assert by_id["m4"].link_status == LinkStatus.nil
    assert by_id["m4"].coreference is not None
    assert by_id["m4"].coreference.resolved_from == "m3"


def test_person_coreference_links_generic_person_phrase(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_LI_AN",
                canonical_name="李安",
                entity_type=EntityType.PERSON,
                aliases=[],
                description="华人电影导演。",
                keywords=["李安", "导演", "电影"],
            )
        ],
    )
    service.config.coreference_terms = {"这个人"}
    request = LinkRequest(
        request_id="score-person-coref",
        text={"content": "李安执导了多部电影，这个人也获得过奥斯卡。", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "李安", "start_offset": 0, "end_offset": 2},
            {"mention_id": "m2", "surface_form": "这个人", "start_offset": 10, "end_offset": 13},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    by_id = {item.mention_id: item for item in response.results}
    assert by_id["m2"].link_status == LinkStatus.linked
    assert by_id["m2"].entity is not None
    assert by_id["m2"].entity.entity_id == "E_LI_AN"
    assert by_id["m2"].coreference is not None
    assert by_id["m2"].coreference.resolved_from == "m1"


def test_cataphora_pronouns_resolve_to_later_named_entity(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_NEZHA",
                canonical_name="哪吒",
                entity_type=EntityType.PERSON,
                aliases=["小哪吒"],
                description="神话人物和动画英雄。",
                keywords=["小哪吒", "英雄", "哪吒"],
            )
        ],
    )
    service.config.coreference_terms = {"他"}
    text = "是他，就是他，我们的英雄，小哪吒！"
    request = LinkRequest(
        request_id="score-cataphora-nezha",
        text={"content": text, "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "他", "start_offset": 1, "end_offset": 2},
            {"mention_id": "m2", "surface_form": "他", "start_offset": 5, "end_offset": 6},
            {"mention_id": "m3", "surface_form": "我们的英雄", "start_offset": 7, "end_offset": 12},
            {"mention_id": "m4", "surface_form": "小哪吒", "start_offset": 13, "end_offset": 16},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    by_id = {item.mention_id: item for item in response.results}
    assert by_id["m4"].link_status == LinkStatus.linked
    assert by_id["m4"].entity is not None
    assert by_id["m4"].entity.entity_id == "E_NEZHA"
    for mention_id in ("m1", "m2", "m3"):
        assert by_id[mention_id].link_status == LinkStatus.linked
        assert by_id[mention_id].entity is not None
        assert by_id[mention_id].entity.entity_id == "E_NEZHA"
        assert by_id[mention_id].coreference is not None
        assert by_id[mention_id].coreference.resolved_from == "m4"


def test_duplicate_canonical_candidates_can_auto_accept_with_context(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_GREEN_1",
                canonical_name="绿水青山就是金山银山",
                entity_type=EntityType.OTHER,
                aliases=[],
                description="生态文明理念。",
                keywords=["绿水青山就是金山银山", "生态文明"],
            ),
            Entity(
                entity_id="E_GREEN_2",
                canonical_name="绿水青山就是金山银山",
                entity_type=EntityType.OTHER,
                aliases=[],
                description="生态文明理念。",
                keywords=["绿水青山就是金山银山", "生态文明"],
            ),
        ],
    )
    request = LinkRequest(
        request_id="score-duplicate-canonical",
        text={"content": "他提出绿水青山就是金山银山理念，强调生态保护。", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "绿水青山就是金山银山",
                "start_offset": 3,
                "end_offset": 13,
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.canonical_name == "绿水青山就是金山银山"
    assert any("同名重复实体" in evidence.detail for evidence in result.evidence)


def test_context_infers_mention_type_and_exposes_candidate_entity_types(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_LOC",
                canonical_name="西湖",
                entity_type=EntityType.LOC,
                aliases=[],
                description="杭州著名景区。",
                keywords=["杭州", "景区", "湖"],
            ),
            Entity(
                entity_id="E_WORK",
                canonical_name="西湖",
                entity_type=EntityType.OTHER,
                aliases=[],
                description="同名歌曲作品。",
                keywords=["歌曲", "专辑", "演唱"],
            ),
        ],
    )
    request = LinkRequest(
        request_id="score-mention-type-context",
        text={"content": "杭州西湖景区游客很多", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "西湖",
                "start_offset": 2,
                "end_offset": 4,
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.mention_type.value == "LOC"
    assert result.entity is not None
    assert result.entity.entity_id == "E_LOC"
    assert result.entity.entity_type.value == "LOC"
    assert result.candidates[0].entity_type.value == "LOC"
    assert any("mention 类型识别：LOC" in evidence.detail for evidence in result.evidence)


def test_exact_match_candidate_type_takes_priority_over_context_heuristic(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_GRID",
                canonical_name="国家电网有限公司",
                entity_type=EntityType.ORG,
                aliases=["国网"],
                description="全国电网运营企业。",
                keywords=["特高压", "输电", "电网"],
            ),
        ],
    )
    request = LinkRequest(
        request_id="score-mention-type-exact-priority",
        text={"content": "国网近期推进跨省特高压线路扩容。", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "国网",
                "start_offset": 0,
                "end_offset": 2,
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    prepared, diagnostics = service.prepare_request(request)

    assert prepared.mentions[0].mention_type.value == "ORG"
    assert diagnostics["m1"]["status"] == "exact_match"
    assert diagnostics["m1"]["mention_type"] == "ORG"


def test_route_mentions_splits_direct_nil_and_coreference_pending(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E001",
                canonical_name="State Grid Corporation of China",
                entity_type=EntityType.ORG,
                aliases=["SGCC"],
                description="National grid operator.",
                keywords=["grid", "power"],
            ),
        ],
    )
    service.config.coreference_terms = {"it"}
    request = LinkRequest(
        request_id="route-buckets",
        text={"content": "SGCC expanded projects and it kept investing.", "language": "en"},
        mentions=[
            {"mention_id": "m1", "surface_form": "SGCC", "start_offset": 0, "end_offset": 4},
            {"mention_id": "m2", "surface_form": "UnknownCo", "start_offset": 5, "end_offset": 14},
            {"mention_id": "m3", "surface_form": "it", "start_offset": 27, "end_offset": 29},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    prepared, mention_type_diagnostics = service.prepare_request(request)
    state = {
        "request": prepared,
        "mention_type_diagnostics": mention_type_diagnostics,
        **service._load_kb({"request": prepared}),
    }
    state.update(service._generate_candidates(state))
    routed = service._route_mentions(state)

    assert routed["mention_routes"]["m1"] == candidate_mod.ROUTE_DIRECT_LINK
    assert routed["mention_routes"]["m2"] == candidate_mod.ROUTE_NIL_PENDING
    assert routed["mention_routes"]["m3"] == candidate_mod.ROUTE_COREFERENCE_PENDING
    assert routed["routing_buckets"][candidate_mod.ROUTE_DIRECT_LINK] == ["m1"]
    assert routed["routing_buckets"][candidate_mod.ROUTE_NIL_PENDING] == ["m2"]
    assert routed["routing_buckets"][candidate_mod.ROUTE_COREFERENCE_PENDING] == ["m3"]
