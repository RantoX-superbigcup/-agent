# 数据规范

## 1. Mention 输入格式

```json
{
  "mention_id": "m1",
  "text": "国网",
  "start": 0,
  "end": 2,
  "entity_type": "organization",
  "sentence": "国网表示将推进配电数字化。",
  "metadata": {
    "coreference_hint": false
  }
}
```

字段说明：

- `mention_id`: 当前指称唯一标识。
- `text`: mention 原文。
- `start` / `end`: 可选，字符位置。
- `entity_type`: 可选，类型先验。
- `sentence`: 可选，包含 mention 的局部上下文。
- `metadata`: 扩展字段，例如 `coreference_hint`。

## 2. 知识库实体格式

```json
{
  "entity_id": "org:sgcc",
  "canonical_name": "国家电网有限公司",
  "aliases": ["国家电网", "国网", "国网公司"],
  "entity_type": "organization",
  "keywords": ["电网", "配电", "输电", "供电"],
  "metadata": {
    "domain": "energy"
  }
}
```

字段说明：

- `entity_id`: 唯一主键。
- `canonical_name`: 标准全称。
- `aliases`: 别名、简称、曾用名。
- `entity_type`: 实体类型。
- `keywords`: 用于上下文消歧的辅助关键词。
- `metadata`: 行业、层级、上位实体等扩展信息。

## 3. 评测样本格式

`data/examples/sample_benchmark.json` 采用如下结构：

```json
{
  "knowledge_base_id": "sample-energy-v1",
  "documents": [
    {
      "doc_id": "doc-001",
      "text": "......",
      "mentions": [],
      "gold": [
        {
          "mention_id": "m1",
          "entity_id": "org:sgcc",
          "status": "linked",
          "is_alias": true
        }
      ]
    }
  ]
}
```

其中：

- `status` 取值为 `linked` 或 `nil`。
- `is_alias` 用于估算别名标准化召回率。
