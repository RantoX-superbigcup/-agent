# LangChain + LangGraph 工作流说明

## 设计目标

课题 10 的流程天然适合图编排。当前工程包含两层 LangGraph：对话理解工作流负责把自然语言用户输入转换为结构化 action；实体链接工作流负责输入校验、知识库加载、候选召回、候选重排、链接决策、人工复核、响应构造和 trace 持久化。

## 对话理解图结构

```text
START
  -> llm_understand
  -> route_after_llm
       accepted -> finalize_action
       fallback -> rule_fallback
  -> END
```

- `llm_understand`: 调用 DeepSeek 解析用户意图、知识库、文本、mention 和运行请求。
- `finalize_action`: 接受高置信 LLM 结果，形成结构化 action。
- `rule_fallback`: DeepSeek 未配置、调用失败或低置信时，退回本地规则解析。

## 实体链接图结构

```text
START
  -> validate_input
  -> input_route
       ok -> load_kb
       invalid -> build_response
  -> load_kb
  -> generate_candidates
  -> candidate_route
       has_candidates -> rerank_candidates
       empty_candidates -> nil_fallback
  -> nil_fallback -> build_response
  -> rerank_candidates
  -> resolve_mentions
  -> review_route
       auto_accept -> build_response
       needs_review -> human_review
  -> human_review
  -> build_response
  -> persist_trace
  -> END
```

## 节点说明

- `validate_input`: 校验文本、mention、span、候选数量和阈值配置。非法输入直接短路到响应构造。
- `load_kb`: 加载内置知识库或请求内联知识库。
- `generate_candidates`: 根据 mention 的标准名、别名、简称召回候选。
- `candidate_route`: 如果全部 mention 都没有候选，则进入 `nil_fallback`。
- `nil_fallback`: 为无候选 mention 直接构造 NIL 结果，并保留解释证据。
- `rerank_candidates`: 结合上下文关键词和实体类型对候选重排。
- `resolve_mentions`: 输出 `linked`、`ambiguous` 或 `nil`，并做简单共指回链。
- `review_route`: 低置信、`ambiguous` 或需要人工确认的结果进入复核分支。
- `human_review`: 标记待复核结果，给出 `human_review_required` 证据。
- `build_response`: 构造对外响应、汇总结果和决策日志。
- `persist_trace`: 持久化 trace，支持后续审计和回放。

## LangChain 使用点

每个 LangGraph 节点都用 LangChain 的 `RunnableLambda` 封装。这样做的好处是后续可以把节点替换为 LangChain retriever、LLM chain、reranker 或本地模型调用，而 FastAPI 接口保持稳定。

## 代码入口

- 图定义：`src/entity_linking_agent/core/workflow.py`
- 服务调用：`src/entity_linking_agent/core/service.py`
- HTTP 接口：`src/entity_linking_agent/api/routes.py`

## 后续可扩展方向

- 把 `generate_candidates` 替换为 LangChain retriever + 向量库。
- 把 `rerank_candidates` 替换为 cross-encoder 或 LLM 判别链。
- 把 `human_review` 替换为真实工单系统、命令行确认或前端审核台。
- 为 LangGraph 增加 checkpointer，支持更细粒度的断点恢复。

## CCKS2019-EL 接入

当前工程增加了 `kb/ccks2019.py` 作为数据适配层：

- `kb_data` 转换为内部 `KnowledgeBaseEntity`。
- `train.json` 中的 `mention_data` 转换为 `MentionRecord`。
- `develop.json` 没有标注 mention，适合作为预测集；如果要直接跑 develop，需要先加实体识别或别名词典匹配节点。

为了避免大知识库下逐实体扫描，候选召回层已经加入别名索引：mention 能精确命中 subject 或 alias 时，会直接返回候选实体。
