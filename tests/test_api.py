import pytest
import json


_ENTITIES = [
    {
        "entity_id": "E001",
        "canonical_name": "国网江苏省电力有限公司",
        "entity_type": "ORG",
        "aliases": ["国网江苏电力", "江苏电力"],
        "former_names": [],
        "description": "国家电网有限公司在江苏地区的省级电力公司。",
        "keywords": ["江苏", "南京", "电力"],
    }
]


@pytest.fixture(scope="module", autouse=True)
def setup_kb(client, kb_id):
    r = client.post("/api/v1/knowledge-bases", json={
        "kb_id": kb_id,
        "kb_version": "v1",
        "description": "fixture kb",
        "entities": _ENTITIES,
    })
    assert r.status_code == 201


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_import_kb_overwrite(client, kb_id):
    """重复导入同一 kb_id 应覆盖，不报 409。"""
    r = client.post("/api/v1/knowledge-bases", json={
        "kb_id": kb_id,
        "kb_version": "v1",
        "description": "overwritten",
        "entities": _ENTITIES,
    })
    assert r.status_code == 201
    assert r.json()["status"] == "overwritten"


def test_import_kb_accepts_nested_package_in_entities(client):
    """兼容前端误把完整 KBPackage 塞进 entities 字段的手动导入请求。"""
    r = client.post("/api/v1/knowledge-bases", json={
        "kb_id": "nested-wrapper-kb",
        "kb_version": "v1",
        "description": "outer description",
        "entities": {
            "kb_id": "inner-kb",
            "kb_version": "v1",
            "description": "inner description",
            "entities": _ENTITIES,
        },
    })

    assert r.status_code == 201
    assert r.json()["kb_id"] == "nested-wrapper-kb"
    assert r.json()["entity_count"] == 1


def test_import_kb_empty_entities(client):
    """entities 为空数组应被 Pydantic 拦截。"""
    r = client.post("/api/v1/knowledge-bases", json={
        "kb_id": "empty-kb",
        "kb_version": "v1",
        "description": "",
        "entities": [],
    })
    assert r.status_code == 422  # Pydantic validation error


def test_import_kb_duplicate_ids(client):
    """同批次 entity_id 重复应拒绝。"""
    r = client.post("/api/v1/knowledge-bases", json={
        "kb_id": "dup-kb",
        "kb_version": "v1",
        "description": "",
        "entities": [
            {"entity_id": "E001", "canonical_name": "A", "entity_type": "ORG",
             "aliases": [], "former_names": [], "description": ""},
            {"entity_id": "E001", "canonical_name": "B", "entity_type": "ORG",
             "aliases": [], "former_names": [], "description": ""},
        ],
    })
    assert r.status_code == 422  # Pydantic model_validator


def test_list_kbs(client, kb_id):
    r = client.get("/api/v1/knowledge-bases")
    assert r.status_code == 200
    ids = [kb["kb_id"] for kb in r.json()["knowledge_bases"]]
    assert kb_id in ids


def test_get_kb(client, kb_id):
    r = client.get(f"/api/v1/knowledge-bases/{kb_id}")
    assert r.status_code == 200
    assert r.json()["kb_id"] == kb_id
    assert r.json()["entity_count"] == 1


def test_import_kb_from_ccks_kb_data_file(client, tmp_path):
    kb_file = tmp_path / "kb_data.jsonl"
    records = [
        {
            "subject_id": "349056",
            "subject": "李安",
            "alias": ["Ang Lee", "李安导演"],
            "type": ["人物"],
            "data": [
                {"predicate": "摘要", "object": "李安，华人电影导演，代表作包括《断背山》。"},
                {"predicate": "职业", "object": "导演"},
            ],
        },
        {
            "subject_id": "83393",
            "subject": "断背山",
            "alias": ["Brokeback Mountain"],
            "type": ["作品"],
            "data": [
                {"predicate": "摘要", "object": "《断背山》是一部由李安执导的电影。"},
            ],
        },
    ]
    kb_file.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")

    r = client.post("/api/v1/knowledge-bases/import-file", json={
        "file_path": str(kb_file),
        "kb_id": "ccks-import-test",
        "kb_version": "v1",
        "source_type": "ccks_kb_data",
        "import_to_store": True,
        "preview_limit": 2,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in {"created", "overwritten"}
    assert data["entity_count"] == 2
    assert data["entities_preview"][0]["canonical_name"] == "李安"

    r = client.get("/api/v1/knowledge-bases/ccks-import-test")
    assert r.status_code == 200
    assert r.json()["entity_count"] == 2


def test_import_kb_from_extensionless_ccks_kb_data_auto(client, tmp_path):
    kb_file = tmp_path / "kb_data"
    records = [
        {
            "subject_id": "349056",
            "subject": "\u674e\u5b89",
            "alias": ["Ang Lee", "\u674e\u5b89\u5bfc\u6f14"],
            "type": ["\u4eba\u7269"],
            "data": [
                {"predicate": "\u6458\u8981", "object": "\u674e\u5b89\uff0c\u534e\u4eba\u7535\u5f71\u5bfc\u6f14\uff0c\u4ee3\u8868\u4f5c\u5305\u62ec\u300a\u65ad\u80cc\u5c71\u300b\u3002"},
                {"predicate": "\u804c\u4e1a", "object": "\u5bfc\u6f14"},
            ],
        }
    ]
    kb_file.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")

    r = client.post("/api/v1/knowledge-bases/import-file", json={
        "file_path": str(kb_file),
        "kb_id": "ccks-extensionless-test",
        "kb_version": "v1",
        "source_type": "auto",
        "import_to_store": True,
        "preview_limit": 1,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["source_type"] == "ccks_kb_data"
    assert data["entity_count"] == 1
    assert data["entities_preview"][0]["canonical_name"] == "\u674e\u5b89"
    assert data["entities_preview"][0]["entity_type"] == "PERSON"


def test_import_kb_from_wrapped_kb_data_object(client, tmp_path):
    kb_file = tmp_path / "wrapped.json"
    payload = {
        "kb_data": [
            {
                "subject_id": "83393",
                "subject": "\u65ad\u80cc\u5c71",
                "alias": ["Brokeback Mountain"],
                "type": ["\u4f5c\u54c1"],
                "data": [
                    {"predicate": "\u6458\u8981", "object": "\u300a\u65ad\u80cc\u5c71\u300b\u662f\u4e00\u90e8\u7531\u674e\u5b89\u6267\u5bfc\u7684\u7535\u5f71\u3002"},
                ],
            }
        ]
    }
    kb_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    r = client.post("/api/v1/knowledge-bases/import-file", json={
        "file_path": str(kb_file),
        "kb_id": "ccks-wrapped-test",
        "kb_version": "v1",
        "source_type": "auto",
        "import_to_store": True,
        "preview_limit": 1,
    })
    assert r.status_code == 200
    data = r.json()
    assert data["entity_count"] == 1
    assert data["entities_preview"][0]["entity_id"] == "83393"
    assert data["entities_preview"][0]["canonical_name"] == "\u65ad\u80cc\u5c71"


def test_import_kb_from_file_missing_path(client):
    r = client.post("/api/v1/knowledge-bases/import-file", json={
        "file_path": "not-exists-kb-data.json",
        "source_type": "auto",
    })
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "FILE_NOT_FOUND"


def test_import_kb_from_uploaded_ccks_file(client):
    records = [
        {
            "subject_id": "u001",
            "subject": "北京大学",
            "alias": ["北大", "Peking University"],
            "type": ["机构"],
            "data": [
                {"predicate": "摘要", "object": "北京大学是中国著名高等学校。"},
            ],
        }
    ]
    content = "\n".join(json.dumps(item, ensure_ascii=False) for item in records).encode("utf-8")
    r = client.post(
        "/api/v1/knowledge-bases/import-upload",
        data={
            "kb_id": "uploaded-ccks-test",
            "kb_version": "v1",
            "source_type": "ccks_kb_data",
            "import_to_store": "true",
            "preview_limit": "1",
            "use_llm": "false",
        },
        files={"file": ("kb_data.jsonl", content, "application/json")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] in {"created", "overwritten"}
    assert data["entity_count"] == 1
    assert data["entities_preview"][0]["canonical_name"] == "北京大学"


def test_import_kb_from_uploaded_extensionless_kb_data_auto(client):
    records = [
        {
            "subject_id": "u-kb-data",
            "subject": "\u676d\u5dde",
            "alias": ["Hangzhou"],
            "type": ["\u5730\u70b9"],
            "data": [
                {"predicate": "\u6458\u8981", "object": "\u676d\u5dde\u662f\u6d59\u6c5f\u7701\u7701\u4f1a\u57ce\u5e02\u3002"},
            ],
        }
    ]
    content = "\n".join(json.dumps(item, ensure_ascii=False) for item in records).encode("utf-8")
    r = client.post(
        "/api/v1/knowledge-bases/import-upload",
        data={
            "kb_id": "uploaded-extensionless-kb-data-test",
            "kb_version": "v1",
            "source_type": "auto",
            "import_to_store": "true",
            "preview_limit": "1",
            "use_llm": "false",
        },
        files={"file": ("kb_data", content, "application/octet-stream")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["source_type"] == "ccks_kb_data"
    assert data["entity_count"] == 1
    assert data["entities_preview"][0]["canonical_name"] == "\u676d\u5dde"


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
