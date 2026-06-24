# Topic 10 Entity Linking Agent

课题 10「实体链接与知识对齐智能体」框架工程。

这个工程按任务书中的“专项能力智能体 + 通用交付要求”搭建，并使用 LangChain + LangGraph 组织实体链接流程。目标是交付一个可运行、可集成、可评测、可追溯、可私有化部署的实体链接服务。当前版本提供了可工作的基线链路，后续可以继续替换为更强的候选召回、重排序、NIL 检测和共指模型。

## 终端智能体

推荐直接使用对话式终端 Agent 演示课题10能力：

```powershell
cd topic10_entity_linking_agent
.\.venv\Scripts\python -m entity_linking_agent.cli
```

如果要启用 DeepSeek 作为 base model 对话理解层，先设置 API Key：

```powershell
$env:DEEPSEEK_API_KEY = "你的DeepSeek API Key"
$env:DEEPSEEK_MODEL = "deepseek-v4-flash"
.\.venv\Scripts\python -m entity_linking_agent.cli
```

DeepSeek 启用后，Agent 会先用模型理解自然语言、长文本、知识库切换、运行意图和 mention 抽取；如果没有设置 API Key，会自动退回本地规则逻辑。需要强制禁用 LLM 时：

```powershell
.\.venv\Scripts\python -m entity_linking_agent.cli --no-llm
```

DeepSeek 还会辅助做知识库路由和 mention 归一化，例如：

```text
李导演的《断背山》真是令人动人 其中实体是李导演和断背山
```

会优先切到 `ccks2019-v1`，并把 `李导演` 扩展为候选别名 `李安`，最终仍保留原始 mention 展示。

启动后可以多轮对话：

```text
你：换成CCKS知识库
Agent：已切换到 ccks2019-v1。现在请给我短文本和 mention。

你：文本：南京南站:坐高铁在南京南站下。南京南站
Agent：短文本已收到。请告诉我要链接的 mention，用逗号分隔。

你：实体是 南京南站,高铁
Agent：正在执行 LangGraph 实体链接流程...
```

一条命令运行 CCKS2019 演示：

```powershell
.\.venv\Scripts\python -m entity_linking_agent.cli --demo
```

传入自定义短句：

```powershell
.\.venv\Scripts\python -m entity_linking_agent.cli --kb ccks2019-v1 --text "南京南站:坐高铁在南京南站下。南京南站" --mentions "南京南站,高铁"
```

终端会输出标准实体、实体ID、候选、置信度、trace_id 和 LangGraph 节点轨迹。

## 对外暴露接口

服务默认监听 `0.0.0.0:8000`，局域网或服务器内其他系统可以通过机器 IP 访问 `/health`、`/docs`、`/v1/link`、`/v1/link/batch`、`/v1/traces/{trace_id}`。

## 文档索引

- [数据流说明 DOCX](docs/data_flow.docx)：从终端输入、DeepSeek 对话图、实体链接图到最终 trace 的完整字段流转。
- [数据流说明 Markdown](docs/data_flow.md)：DOCX 的可维护源文件。
- [LangChain + LangGraph 工作流说明](docs/langchain_langgraph_workflow.md)：两层 LangGraph 节点和条件边说明。
- [API 规范](docs/api_spec.md)：HTTP 接口、请求字段和响应字段说明。
- [架构说明](docs/architecture.md)：模块职责和设计取舍。

## 对应任务书要求

- 服务化、可集成：基于 FastAPI 对外暴露 HTTP API。
- LangChain + LangGraph：用 `StateGraph` 编排带条件边的 Agent 工作流，包含输入校验、候选为空 NIL 兜底、低置信/歧义人工复核分支。
- DeepSeek base model：可选接入 DeepSeek API 作为终端 Agent 的自然语言理解层，负责解析用户意图、长文本和 mention。
- 可追溯、可回放：每次调用自动生成 `trace_id`，并把链路结果写入 `artifacts/traces/`；权限受限时自动回退到系统临时目录和进程内缓存。
- 可评测、可复现：提供 `scripts/evaluate.py` 和 `data/examples/sample_benchmark.json`。
- 可私有化部署：提供本地公开监听、命令行入口和 `Dockerfile`。
- 建议交付物：代码、接口文档、数据规范、样例数据、评测方案、技术报告模板、badcase 模板均已预留。

## 目录结构

```text
topic10_entity_linking_agent/
|-- src/entity_linking_agent/
|   |-- api/                  # HTTP 路由
|   |-- core/                 # 链接流程、追踪存储、服务编排
|   |-- kb/                   # 知识库加载
|   |-- models/               # API Schema
|   `-- utils/                # 文本与 trace 工具
|-- data/
|   |-- examples/             # 请求样例、基准样例
|   `-- kb/                   # 示例知识库
|-- docs/                     # 架构、接口、数据规范、部署与评测说明
|-- reports/                  # 技术报告与 badcase 模板
|-- scripts/                  # 启动、演示与评测脚本
|-- tests/                    # 单元测试
|-- artifacts/traces/         # trace 落盘目录
|-- Dockerfile
`-- pyproject.toml
```

## 快速启动

```powershell
cd topic10_entity_linking_agent
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python .\main.py
```

如果你在 `(uno)` 这类 Python 3.9 环境里运行，本工程已经把 LangChain/LangGraph 依赖钉在 3.9 可用的 0.3/0.2 系列。

等价的 uvicorn 启动方式：

```powershell
.\.venv\Scripts\python -m uvicorn entity_linking_agent.app:app --app-dir src --host 0.0.0.0 --port 8000
```

或者使用安装后的命令行入口：

```powershell
$env:EL_HOST = "0.0.0.0"
$env:EL_PORT = "8000"
.\.venv\Scripts\topic10-agent.exe
```

## 示例请求

本机测试：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/v1/link" `
  -ContentType "application/json" `
  -InFile ".\data\examples\sample_request.json"
```

局域网或其他机器访问：

```powershell
Invoke-RestMethod `
  -Method Post `
  -Uri "http://<服务器IP>:8000/v1/link" `
  -ContentType "application/json" `
  -InFile ".\data\examples\sample_request.json"
```

## 当前基线能力

- 候选实体生成：基于标准名和别名的轻量召回。
- 上下文消歧：用上下文关键词对候选重打分。
- 实体标准化：输出标准全称 + 唯一 ID。
- 输入校验：空文本、空 mention、非法偏移或异常阈值会在 `validate_input` 节点直接拦截。
- NIL 检测：分数低于阈值时输出 `nil`。
- 候选为空兜底：全部 mention 无候选时走 `nil_fallback`，不再进入无意义重排。
- 人工复核路由：低置信、共指回链或 `ambiguous` 结果会进入 `human_review` 节点。
- 共指兜底：支持 `该公司/该机构` 这类简单共指回链。
- 留痕解释：输出候选分数、命中别名、关键词依据和决策日志。
- LLM 对话增强：检测到 `DEEPSEEK_API_KEY` 时启用 DeepSeek，未配置时自动保持本地规则模式。
- LLM 别名扩展：支持把 `李导演` 这类上下文代称扩展为 `李安` 等候选别名，再交给可解释链接流程决策。

## LangGraph 工作流

终端 Agent 现在包含两层 LangGraph 工作流。

对话理解编排在 `src/entity_linking_agent/core/dialogue_workflow.py`：

```text
START
  -> llm_understand
  -> [accepted] finalize_action
  -> [fallback] rule_fallback
  -> END
```

实体链接核心编排在 `src/entity_linking_agent/core/workflow.py`：

```text
START
  -> validate_input
  -> [invalid] build_response
  -> [ok] load_kb
  -> generate_candidates
  -> [empty_candidates] nil_fallback
  -> [has_candidates] rerank_candidates
  -> resolve_mentions
  -> [needs_review] human_review
  -> [auto_accept] build_response
  -> build_response
  -> persist_trace
  -> END
```

每个节点都用 LangChain 的 `RunnableLambda` 包装，后续可以把候选召回替换成 LangChain retriever，把消歧节点替换成 LLM/reranker，接口层不需要变化。

## CCKS2019-EL 数据集

工程已支持读取同级目录下的 `ccks2019_el` 数据集：

```text
E:\平时\2026实训\阶段2\ccks2019_el
|-- kb_data
|-- train.json
`-- develop.json
```

快速跑一个小规模验证：

```powershell
cd topic10_entity_linking_agent
.\.venv\Scripts\python .\scripts\evaluate_ccks2019.py --max-docs 50
```

`kb_data` 可通过 `knowledge_base_id=ccks2019-v1` 作为完整知识库加载；日常调试更推荐先用 `evaluate_ccks2019.py` 筛选子集，速度更稳。

## 运行评测

```powershell
cd topic10_entity_linking_agent
python .\scripts\evaluate.py .\data\examples\sample_benchmark.json
python -m unittest discover -s tests
```

## 后续建议

- 把 `core/retriever.py` 替换为 BM25、向量检索或混合召回。
- 在 `core/linker.py` 接入 cross-encoder / reranker 做精排与 NIL 判定。
- 将 `data/kb/` 替换为行业词库或图数据库导出的实体库。
- 在 `scripts/evaluate.py` 中接入真实公开数据集，补充准确率、召回率、F1 和 badcase 汇总。
