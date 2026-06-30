from __future__ import annotations

from types import SimpleNamespace

from app.models.entity import Entity
from app.models.enums import EntityType, LinkStatus
from app.models.request import LinkRequest
from app.services.link_service import LinkService
from app.storage.kb_store import KBStore


def _service(tmp_path, entities: list[Entity]) -> LinkService:
    store = KBStore(tmp_path)
    store.import_full("score-kb", "v1", "scoring regression kb", entities)
    config = SimpleNamespace(coreference_terms=set(), index_dir=None)
    return LinkService(store, config)


def test_llm_alias_expansion_beats_misleading_surface_alias(tmp_path):
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
        request_id="score-alias-expansion",
        text={"content": "李导演的《断背山》真是令人动人", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "李导演",
                "start_offset": 0,
                "end_offset": 3,
                "entity_type": "PERSON",
                "candidate_aliases": ["李安"],
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_ANGLEE"
    assert any(e.detail.endswith("llm_alias_expansion") for e in result.evidence)


def test_weak_llm_alias_expansion_requires_review(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_TSUI",
                canonical_name="徐克",
                entity_type=EntityType.PERSON,
                aliases=["徐老怪"],
                description="香港电影导演，执导《智取威虎山》《黄飞鸿》等电影。",
                keywords=["徐克", "导演", "电影", "智取威虎山", "黄飞鸿"],
            )
        ],
    )
    request = LinkRequest(
        request_id="score-weak-alias-expansion",
        text={"content": "徐先生的电影广受好评", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "徐先生",
                "start_offset": 0,
                "end_offset": 3,
                "entity_type": "PERSON",
                "candidate_aliases": ["徐克"],
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.ambiguous
    assert result.entity is not None
    assert result.entity.entity_id == "E_TSUI"
    assert any("human_review_required" in e.detail for e in result.evidence)
    assert not any("别名扩展已通过上下文验证" in e.detail for e in result.evidence)


def test_strong_context_validates_llm_alias_expansion(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_TSUI",
                canonical_name="徐克",
                entity_type=EntityType.PERSON,
                aliases=["徐老怪"],
                description="香港电影导演，执导《智取威虎山》《黄飞鸿》等电影。",
                keywords=["徐克", "导演", "电影", "智取威虎山", "黄飞鸿"],
            )
        ],
    )
    request = LinkRequest(
        request_id="score-strong-alias-expansion",
        text={"content": "徐先生执导的《智取威虎山》广受好评", "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "徐先生",
                "start_offset": 0,
                "end_offset": 3,
                "entity_type": "PERSON",
                "candidate_aliases": ["徐克"],
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_TSUI"
    assert any("别名扩展已通过上下文验证" in e.detail for e in result.evidence)


def test_director_context_disambiguates_brokeback_mountain_movie(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_MOVIE",
                canonical_name="断背山",
                entity_type=EntityType.OTHER,
                aliases=["Brokeback Mountain"],
                description="《断背山》由李安执导，希斯莱杰主演，是一部爱情电影，曾获奥斯卡和金狮奖。",
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


def test_title_context_expands_general_secretary_to_person(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_STARLIGHT",
                canonical_name="星光熠熠",
                entity_type=EntityType.PERSON,
                aliases=["总书记"],
                description="动画角色。",
                keywords=["小马宝莉", "总书记"],
            ),
            Entity(
                entity_id="E_TITLE",
                canonical_name="总书记",
                entity_type=EntityType.OTHER,
                aliases=["第一书记"],
                description="政党最高负责人的称谓。",
                keywords=["总书记", "职务"],
            ),
            Entity(
                entity_id="E_XI",
                canonical_name="习近平",
                entity_type=EntityType.PERSON,
                aliases=[],
                description="现任中国共产党中央委员会总书记，曾提出绿水青山就是金山银山。",
                keywords=["习近平", "总书记", "绿水青山就是金山银山"],
            ),
        ],
    )
    request = LinkRequest(
        request_id="score-title-context",
        text={"content": "杭州西湖的美景离不开总书记的“绿水青山就是金山银山”的方略", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "总书记", "start_offset": 10, "end_offset": 13}
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_XI"


def test_hangzhou_context_expands_west_lake(tmp_path):
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
    request = LinkRequest(
        request_id="score-west-lake-context",
        text={"content": "杭州西湖的美景令人难忘", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "西湖", "start_offset": 2, "end_offset": 4}
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_HANGZHOU_WEST_LAKE"


def test_company_coreference_keeps_nil_antecedent(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_XI",
                canonical_name="习近平",
                entity_type=EntityType.PERSON,
                aliases=[],
                description="现任中国共产党中央委员会总书记，提出绿水青山就是金山银山理念。",
                keywords=["习近平", "总书记", "绿水青山就是金山银山"],
            ),
            Entity(
                entity_id="E_STARLIGHT",
                canonical_name="星光熠熠",
                entity_type=EntityType.PERSON,
                aliases=["总书记"],
                description="动画角色。",
                keywords=["小马宝莉", "总书记"],
            ),
        ],
    )
    service.config.coreference_terms = {"他", "该公司"}
    request = LinkRequest(
        request_id="score-nil-company-coref",
        text={
            "content": "总书记提出绿水青山就是金山银山理念；他强调生态保护。幻影生态科技公司宣布建设景区，但该公司不在知识库中。",
            "language": "zh",
        },
        mentions=[
            {"mention_id": "m1", "surface_form": "总书记", "start_offset": 0, "end_offset": 3},
            {"mention_id": "m2", "surface_form": "他", "start_offset": 19, "end_offset": 20},
            {"mention_id": "m3", "surface_form": "幻影生态科技公司", "start_offset": 27, "end_offset": 35},
            {"mention_id": "m4", "surface_form": "该公司", "start_offset": 45, "end_offset": 48},
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


def test_lake_coreference_links_to_previous_lake_entity(tmp_path):
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


def test_well_coreference_links_generic_facility_phrase(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_MOON_WELL",
                canonical_name="月影井",
                entity_type=EntityType.LOC,
                aliases=[],
                description="古城景区中的一口历史古井。",
                keywords=["月影井", "古井", "景区"],
            )
        ],
    )
    request = LinkRequest(
        request_id="score-well-coref",
        text={"content": "月影井位于古城景区，这口井已有百年历史。", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "月影井", "start_offset": 0, "end_offset": 3},
            {"mention_id": "m2", "surface_form": "这口井", "start_offset": 11, "end_offset": 14},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    by_id = {item.mention_id: item for item in response.results}
    assert by_id["m2"].link_status == LinkStatus.linked
    assert by_id["m2"].entity is not None
    assert by_id["m2"].entity.entity_id == "E_MOON_WELL"
    assert by_id["m2"].coreference is not None
    assert by_id["m2"].coreference.resolved_from == "m1"


def test_deictic_head_coreference_handles_unlisted_location_phrase(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_YANGTZE",
                canonical_name="长江",
                entity_type=EntityType.LOC,
                aliases=[],
                description="中国重要河流。",
                keywords=["长江", "河流"],
            )
        ],
    )
    request = LinkRequest(
        request_id="score-deictic-head-coref",
        text={"content": "长江流经多个省份，这条河孕育了丰富文明。", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "长江", "start_offset": 0, "end_offset": 2},
            {"mention_id": "m2", "surface_form": "这条河", "start_offset": 10, "end_offset": 13},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    by_id = {item.mention_id: item for item in response.results}
    assert by_id["m2"].link_status == LinkStatus.linked
    assert by_id["m2"].entity is not None
    assert by_id["m2"].entity.entity_id == "E_YANGTZE"
    assert by_id["m2"].coreference is not None
    assert by_id["m2"].coreference.resolved_from == "m1"


def test_appositive_titles_corefer_to_previous_entity(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_TARZAN",
                canonical_name="泰山",
                entity_type=EntityType.PERSON,
                aliases=["Tarzan"],
                description="漂泊在丛林中的冒险人物，被称为丛林的霸主。",
                keywords=["泰山", "丛林", "霸主", "漂泊"],
            )
        ],
    )
    request = LinkRequest(
        request_id="score-appositive-coref",
        text={"content": "泰山，丛林的霸主，漂泊之人，今天第一次接触到了城市。", "language": "zh"},
        mentions=[
            {"mention_id": "m1", "surface_form": "泰山", "start_offset": 0, "end_offset": 2},
            {"mention_id": "m2", "surface_form": "丛林的霸主", "start_offset": 3, "end_offset": 8},
            {"mention_id": "m3", "surface_form": "漂泊之人", "start_offset": 9, "end_offset": 13},
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    by_id = {item.mention_id: item for item in response.results}
    assert by_id["m1"].link_status == LinkStatus.linked
    assert by_id["m1"].entity is not None
    assert by_id["m1"].entity.entity_id == "E_TARZAN"
    assert by_id["m2"].link_status == LinkStatus.linked
    assert by_id["m2"].entity is not None
    assert by_id["m2"].entity.entity_id == "E_TARZAN"
    assert by_id["m2"].coreference is not None
    assert by_id["m2"].coreference.resolved_from == "m1"
    assert by_id["m3"].link_status == LinkStatus.linked
    assert by_id["m3"].entity is not None
    assert by_id["m3"].entity.entity_id == "E_TARZAN"
    assert by_id["m3"].coreference is not None
    assert by_id["m3"].coreference.resolved_from == "m2"


def test_descriptive_reference_retrieves_entity_from_context_description(tmp_path):
    service = _service(
        tmp_path,
        [
            Entity(
                entity_id="E_DUNHUANG",
                canonical_name="敦煌",
                entity_type=EntityType.LOC,
                aliases=[],
                description="敦煌是莫高窟所在地，是以壁画飞天闻名的丝路重镇。",
                keywords=["敦煌", "莫高窟", "丝路重镇", "壁画", "飞天"],
            ),
            Entity(
                entity_id="E_PINGYAO",
                canonical_name="平遥",
                entity_type=EntityType.LOC,
                aliases=[],
                description="平遥是晋商票号发源地，古城墙保存完好。",
                keywords=["平遥", "晋商票号", "古城墙"],
            ),
        ],
    )
    text = "莫高窟所在的那座丝路重镇以壁画飞天闻名。"
    request = LinkRequest(
        request_id="score-descriptive-reference",
        text={"content": text, "language": "zh"},
        mentions=[
            {
                "mention_id": "m1",
                "surface_form": "那座丝路重镇",
                "start_offset": text.find("那座丝路重镇"),
                "end_offset": text.find("那座丝路重镇") + len("那座丝路重镇"),
            }
        ],
        knowledge_base={"kb_id": "score-kb", "kb_version": "v1"},
        options={"top_k": 5, "nil_threshold": 0.6, "ambiguity_margin": 0.08},
    )

    response = service.link(request)

    result = response.results[0]
    assert result.link_status == LinkStatus.linked
    assert result.entity is not None
    assert result.entity.entity_id == "E_DUNHUANG"
    assert any("描述性指称" in evidence.detail for evidence in result.evidence)


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
    assert response.coreference_chains
    assert response.coreference_chains[0].mention_ids == ["m1", "m2", "m3", "m4"]


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
