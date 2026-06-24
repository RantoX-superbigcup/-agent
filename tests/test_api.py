import pytest


@pytest.fixture(scope="module", autouse=True)
def setup_kb(client, kb_id):
    client.post("/api/v1/knowledge-bases", json={
        "kb_id": kb_id, "kb_version": "v1", "description": "fixture kb"
    })
    client.post(f"/api/v1/knowledge-bases/{kb_id}/entities", json={"entities": [
        {
            "entity_id": "E001",
            "canonical_name": "国网江苏省电力有限公司",
            "entity_type": "ORG",
            "aliases": ["国网江苏电力", "江苏电力"],
            "former_names": [],
            "description": "国家电网有限公司在江苏地区的省级电力公司。",
            "keywords": ["江苏", "南京", "电力"],
        }
    ]})


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_create_kb_duplicate(client, kb_id):
    r = client.post("/api/v1/knowledge-bases", json={
        "kb_id": kb_id, "kb_version": "v1", "description": ""
    })
    assert r.status_code == 409


def test_list_kbs(client, kb_id):
    r = client.get("/api/v1/knowledge-bases")
    assert r.status_code == 200
    ids = [kb["kb_id"] for kb in r.json()["knowledge_bases"]]
    assert kb_id in ids


def test_get_kb(client, kb_id):
    r = client.get(f"/api/v1/knowledge-bases/{kb_id}")
    assert r.status_code == 200
    assert r.json()["kb_id"] == kb_id


def test_entity_link_alias(client, kb_id):
    r = client.post("/api/v1/entity-link", json={
        "schema_version": "v1",
        "request_id": "req-alias",
        "text": {"content": "国网江苏电力完成南京供电保障。"},
        "mentions": [{"mention_id": "m1", "surface_form": "国网江苏电力", "start_offset": 0, "end_offset": 6}],
        "knowledge_base": {"kb_id": kb_id, "kb_version": "v1"},
    })
    assert r.status_code == 200
    result = r.json()["results"][0]
    assert result["link_status"] == "linked"
    assert result["entity"]["entity_id"] == "E001"


def test_entity_link_nil(client, kb_id):
    r = client.post("/api/v1/entity-link", json={
        "schema_version": "v1",
        "request_id": "req-nil",
        "text": {"content": "这是一个完全没有对应实体的词语。"},
        "mentions": [{"mention_id": "m1", "surface_form": "xyzxyzxyz", "start_offset": 0, "end_offset": 9}],
        "knowledge_base": {"kb_id": kb_id, "kb_version": "v1"},
    })
    assert r.status_code == 200
    assert r.json()["results"][0]["link_status"] == "nil"


def test_entity_link_coreference(client, kb_id):
    r = client.post("/api/v1/entity-link", json={
        "schema_version": "v1",
        "request_id": "req-coref",
        "text": {"content": "国网江苏电力完成任务。该公司表示满意。"},
        "mentions": [
            {"mention_id": "m1", "surface_form": "国网江苏电力", "start_offset": 0, "end_offset": 6},
            {"mention_id": "m2", "surface_form": "该公司", "start_offset": 11, "end_offset": 14},
        ],
        "knowledge_base": {"kb_id": kb_id, "kb_version": "v1"},
    })
    assert r.status_code == 200
    data = r.json()
    results = {item["mention_id"]: item for item in data["results"]}
    assert results["m2"]["link_status"] == "linked"
    assert results["m2"]["coreference"]["resolved_from"] == "m1"


def test_entity_link_kb_not_found(client):
    r = client.post("/api/v1/entity-link", json={
        "schema_version": "v1",
        "request_id": "req-missing",
        "text": {"content": "测试文本"},
        "mentions": [{"mention_id": "m1", "surface_form": "测试", "start_offset": 0, "end_offset": 2}],
        "knowledge_base": {"kb_id": "no-such-kb", "kb_version": "v1"},
    })
    assert r.status_code == 400
