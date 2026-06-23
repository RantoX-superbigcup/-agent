# 任务书映射说明

## 课题 10 本体要求

| 任务书要求 | 当前工程落点 |
| --- | --- |
| 文本 + mention + 知识库输入 | `models/schema.py` 中 `LinkRequestView` |
| 输出标准实体、NIL、共指、依据 | `core/linker.py` + `models/schema.py` |
| 作为归一/清洗环节可调用服务 | `api/routes.py` 暴露标准 HTTP 接口 |
| 链接 / 消歧 / 标准化，不重复做 NER | 输入要求显式为“已识别 mention” |
| 候选生成、上下文消歧、NIL、共指 | `core/retriever.py`、`core/linker.py` |
| LangChain + LangGraph 方法 | `core/workflow.py` 使用 `StateGraph` + `RunnableLambda` 编排节点 |

## 通用交付要求映射

| 通用要求 | 当前工程落点 |
| --- | --- |
| 服务化、可集成 | FastAPI 服务，`/v1/link`、`/v1/link/batch` |
| 输入输出规范清晰稳定 | `docs/api_spec.md`、`docs/data_spec.md` |
| 可追溯、可回放 | `trace_id` + LangGraph `node_events` + `artifacts/traces/` + `/v1/traces/{trace_id}` |
| 可评测、可复现 | `scripts/evaluate.py`、`data/examples/sample_benchmark.json`、`tests/` |
| 可私有化部署 | `Dockerfile`、`docs/deployment.md` |
| 典型样例与失败案例 | `data/examples/`、`reports/badcase_template.md` |
| 技术报告 | `reports/technical_report_outline.md` |

## 目前是“框架工程”而不是“最终算法”

这个版本已经满足课题启动和后续扩展所需的工程骨架，但算法部分仍是基线实现，适合在此基础上继续研究：

- 候选召回目前是别名匹配基线，后续可替换为混合检索。
- 消歧目前是关键词重打分基线，后续可加入向量表征或 reranker。
- 共指与 NIL 仍偏规则化，后续可加入单独模型或判别头。
