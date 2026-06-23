# 架构说明

## 目标

围绕课题 10 的输入输出要求，构建一个可被治理流水线直接调用的实体链接服务，并保留后续研究替换空间。

## 组件

- `api/routes.py`
  负责 HTTP 接口暴露。
- `models/schema.py`
  负责输入输出 Schema 校验。
- `core/workflow.py`
  使用 LangGraph `StateGraph` 编排实体链接流程，并用 LangChain `RunnableLambda` 封装每个图节点。
- `core/retriever.py`
  负责候选实体生成。
- `core/linker.py`
  负责候选重打分、消歧、NIL 与共指兜底。
- `core/service.py`
  负责知识库加载、trace 生成、响应汇总。
- `core/trace_store.py`
  负责链路结果落盘与回放。
- `llm/deepseek_client.py`
  可选接入 DeepSeek base model，负责自然语言意图解析、长文本理解、mention 抽取、知识库路由和 mention 别名扩展。
- `kb/loader.py`
  负责内置或内联知识库加载。

## 处理流程

```text
Request
  -> Schema validation
  -> LangGraph workflow
       validate_input
       load_kb
       generate_candidates
       candidate_route
       nil_fallback
       rerank_candidates
       resolve_mentions
       review_route
       human_review
       build_response
       persist_trace
  -> Response
```

终端对话入口会先经过可选 LLM 层：

```text
User turn
  -> DeepSeek intent parser
  -> dialogue state update
  -> LangGraph entity linking workflow
```

如果未配置 `DEEPSEEK_API_KEY` 或 API 调用失败，终端 Agent 会自动退回本地规则解析，保证离线环境仍能演示。DeepSeek 返回的 `mention_aliases` 会作为候选召回扩展信号，例如把 `李导演` 扩展为 `李安`，但最终实体选择仍由可追踪的实体链接工作流完成。

## 设计取舍

- 任务书强调“建立在 NER 之后”，因此核心服务不强依赖实体识别；终端 Agent 在配置 DeepSeek 后可以辅助抽取 mention。
- 任务书强调“可追溯”，因此每次请求都生成 `trace_id` 并持久化结果。
- 任务书强调“服务化、可集成”，因此接口与数据规范优先于算法复杂度。
- 任务书强调“可评测、可复现”，因此把样例 benchmark 和评测脚本纳入工程本体。
- 使用 LangGraph 的原因是流程天然是可追踪的有向图，后续可以在节点级插入检索器、LLM、人工复核或条件分支。

## 后续扩展点

- 在候选召回层加入 BM25、向量库、图谱邻居检索。
- 在精排层加入 cross-encoder、指称编码器和 NIL 判别器。
- `workflow.py` 已包含条件边：输入非法短路、候选全空 NIL 兜底、低置信或歧义结果进入人工复核节点。
- 在 trace 层加入数据库或消息队列，而非本地文件。
- 在评测层接入公开数据集和行业标注集。
