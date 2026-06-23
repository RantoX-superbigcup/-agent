# API 规范

## 公开访问基地址

服务启动时应监听 `0.0.0.0`，供外部系统通过真实机器 IP 调用：

```text
http://<服务器IP>:8000
```

例如：

```text
http://192.168.1.10:8000/v1/link
```

`0.0.0.0` 只用于服务端监听，客户端请求里不要把它当作目标 IP。

## 1. `GET /health`

健康检查与运行信息。

响应示例：

```json
{
  "status": "ok",
  "service": "Topic 10 Entity Linking Agent",
  "version": "0.1.0",
  "builtin_knowledge_bases": {
    "sample-energy-v1": "E:/.../topic10_entity_linking_agent/data/kb/sample_kb.json"
  },
  "traces_dir": "E:/.../topic10_entity_linking_agent/artifacts/traces"
}
```

## 2. `POST /v1/link`

单文档实体链接。

请求字段：

- `text`: 原始文本。
- `mentions`: 已识别实体指称列表。
- `knowledge_base_id`: 内置知识库 ID。
- `inline_knowledge_base`: 可选，直接内联知识库。
- `options`: 阈值、候选数、共指开关等参数。
- `trace_id`: 可选，自定义追踪 ID。

请求示例见 [sample_request.json](../data/examples/sample_request.json)。

响应核心字段：

- `trace_id`: 本次调用的追踪 ID。
- `kb_id` / `kb_version`: 知识库标识。
- `workflow_engine`: 当前为 `langgraph`。
- `graph_nodes`: 本次执行的 LangGraph 节点列表。
- `route_decision`: 本次最终路由，例如 `resolved`、`empty_candidates`、`needs_review`、`invalid_input`。
- `validation_errors`: 输入校验失败原因；正常请求为空数组。
- `node_events`: 节点级执行摘要，便于审计和回放。
- `results`: 每个 mention 的链接结果。
- `summary`: 链接结果统计。
- `decision_log`: 适合平台侧落库的简要决策日志。

单个 `result` 包含：

- `linked_entity_id`: 链接到的唯一实体 ID。
- `canonical_name`: 标准实体全称。
- `status`: `linked` / `ambiguous` / `nil`。
- `confidence`: 当前基线打分。
- `needs_review`: 是否建议人工复核。
- `candidates`: 候选实体与打分依据。
- `evidence`: 匹配别名、上下文片段、关键词证据。
- `coreference_source_mention_id`: 若为共指回链，记录来源 mention。

## 3. `POST /v1/link/batch`

批量处理多个请求对象。

请求体：

```json
{
  "items": [
    {
      "...": "same as /v1/link request"
    }
  ]
}
```

## 4. `GET /v1/traces/{trace_id}`

按 `trace_id` 获取落盘的完整追踪记录，用于审计、回放或 badcase 复盘。
