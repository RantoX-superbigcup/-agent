# 数据流说明

本文档说明课题 10「实体链接与知识对齐智能体」从用户输入到最终链接结果的完整数据流。工程当前包含两层 LangGraph 工作流：

- 对话理解工作流：把自然语言用户输入转换为结构化 action。
- 实体链接工作流：把结构化请求转换为实体链接、NIL、共指、复核和 trace 结果。

## 一、整体数据流

终端模式的完整链路如下：

```text
用户自然语言输入
  -> ConversationalAgent
  -> DialogueWorkflow
       llm_understand
       finalize_action / rule_fallback
  -> DialogueState
  -> MentionRecord 构造
  -> 知识库选择与候选子集加载
  -> Topic10EntityLinkingService
  -> EntityLinkingWorkflow
       validate_input
       load_kb
       generate_candidates
       candidate_route
       nil_fallback / rerank_candidates
       resolve_mentions
       review_route
       human_review / build_response
       persist_trace
  -> CLI 输出 / API 响应 / trace 记录
```

API 模式的链路更短：

```text
HTTP JSON 请求
  -> LinkRequestView
  -> Topic10EntityLinkingService
  -> EntityLinkingWorkflow
  -> LinkResponseView
```

也就是说：

```text
终端模式 = 对话理解工作流 + 实体链接工作流
API 模式 = 实体链接工作流
```

## 二、核心文件索引

| 模块 | 文件 | 作用 |
| --- | --- | --- |
| 终端入口 | `src/entity_linking_agent/cli.py` | 多轮对话、状态保存、命令行输出 |
| 对话工作流 | `src/entity_linking_agent/core/dialogue_workflow.py` | DeepSeek / 规则兜底的 LangGraph 对话图 |
| DeepSeek 客户端 | `src/entity_linking_agent/llm/deepseek_client.py` | 调用 DeepSeek API，解析自然语言为结构化 action |
| 服务入口 | `src/entity_linking_agent/core/service.py` | 构造 trace、默认参数、调用实体链接工作流 |
| 实体链接工作流 | `src/entity_linking_agent/core/workflow.py` | LangGraph 实体链接主流程 |
| 候选召回 | `src/entity_linking_agent/core/retriever.py` | 根据 mention、别名、LLM 扩展别名召回候选 |
| 消歧与共指 | `src/entity_linking_agent/core/linker.py` | 重打分、NIL、歧义判定、共指回链 |
| 知识库加载 | `src/entity_linking_agent/kb/loader.py` | 加载内置知识库或 inline 知识库 |
| CCKS 适配 | `src/entity_linking_agent/kb/ccks2019.py` | 将 CCKS2019 kb_data 转换为内部实体格式 |
| API Schema | `src/entity_linking_agent/models/schema.py` | 请求和响应字段定义 |
| Trace 存储 | `src/entity_linking_agent/core/trace_store.py` | 持久化或缓存完整执行结果 |

## 三、关键数据结构

### 1. DialogueState

终端 Agent 保存的多轮对话状态。

```python
DialogueState(
    kb_id="sample-energy-v1",
    text="",
    mention_texts=[],
    mention_aliases={}
)
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `kb_id` | 当前使用的知识库，例如 `sample-energy-v1` 或 `ccks2019-v1` |
| `text` | 当前待链接文本 |
| `mention_texts` | 用户指定或 DeepSeek 抽取出的 mention 文本 |
| `mention_aliases` | DeepSeek 生成的候选别名扩展，例如 `{"李导演": ["李安"]}` |

### 2. MentionRecord

实体链接工作流内部使用的 mention 对象。

```python
MentionRecord(
    mention_id="m1",
    text="李导演",
    start=0,
    end=3,
    entity_type=None,
    sentence="李导演的《断背山》真是令人动人",
    metadata={"candidate_aliases": ["李安"]}
)
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `mention_id` | mention 的唯一编号 |
| `text` | 原始 mention，不会被 LLM 改写 |
| `start` / `end` | mention 在文本中的位置 |
| `entity_type` | 可选实体类型 |
| `sentence` | 当前句子或全文上下文 |
| `metadata.candidate_aliases` | DeepSeek 或规则提供的候选别名扩展 |

### 3. KnowledgeBaseEntity

知识库实体对象。

```python
KnowledgeBaseEntity(
    entity_id="349056",
    canonical_name="李安",
    aliases=["Ang Lee", "李安", "ang lee"],
    entity_type="Human",
    keywords=["导演", "断背山", "电影"],
    metadata={"description": "..."}
)
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `entity_id` | 知识库唯一实体 ID |
| `canonical_name` | 标准实体名 |
| `aliases` | 别名列表 |
| `entity_type` | 实体类型 |
| `keywords` | 用于上下文消歧的关键词 |
| `metadata.description` | 实体描述文本，用于重排打分 |

### 4. CandidateScore

候选召回和重排阶段使用的候选对象。

```python
CandidateScore(
    entity_id="349056",
    canonical_name="李安",
    entity_type="Human",
    score=0.872,
    alias_similarity=1.0,
    matched_alias="李安",
    overlapping_keywords=["断背山"],
    reasons=["llm_alias_expansion", "description_overlap_support"]
)
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `score` | 当前候选分数 |
| `alias_similarity` | mention 或扩展别名与知识库别名的相似度 |
| `matched_alias` | 命中的知识库别名 |
| `overlapping_keywords` | 上下文命中的关键词 |
| `reasons` | 可解释原因，例如 `exact_or_alias_match`、`llm_alias_expansion` |

### 5. LinkDecision

最终单个 mention 的链接决策。

```python
LinkDecision(
    mention_id="m1",
    text="李导演",
    linked_entity_id="349056",
    canonical_name="李安",
    status="linked",
    confidence=0.872,
    needs_review=False,
    candidates=[...],
    evidence=EvidenceRecord(...),
    coreference_source_mention_id=None
)
```

`status` 可能取值：

| 状态 | 含义 |
| --- | --- |
| `linked` | 成功链接到知识库实体 |
| `nil` | 知识库中没有合适实体 |
| `ambiguous` | top 候选差距太小，需要复核 |

## 四、对话理解工作流

对话理解图定义在 `DialogueWorkflow`。

```text
START
  -> llm_understand
  -> route_after_llm
       accepted -> finalize_action
       fallback -> rule_fallback
  -> END
```

### 1. 输入

```python
{
    "user_text": "李导演的《断背山》真是令人动人 其中实体是李导演和断背山",
    "current_state": {
        "kb_id": "sample-energy-v1",
        "has_text": False,
        "text_preview": "",
        "mentions": [],
        "mention_aliases": {}
    }
}
```

### 2. llm_understand

如果配置了 `DEEPSEEK_API_KEY`，系统调用 DeepSeek。DeepSeek 的任务不是直接返回实体 ID，而是解析用户意图。

理想输出：

```json
{
  "action": "run",
  "kb_id": "ccks2019-v1",
  "text": "李导演的《断背山》真是令人动人",
  "mentions": ["李导演", "断背山"],
  "mention_aliases": {
    "李导演": ["李安"]
  },
  "run_requested": true,
  "reply": null,
  "confidence": 0.95
}
```

DeepSeek 在这里做四件事：

- 判断用户是否要运行实体链接。
- 抽取待处理文本。
- 抽取 mention 列表。
- 做知识库路由和 mention 别名扩展。

### 3. route_after_llm

如果 DeepSeek 返回：

```text
action != unknown
confidence >= 0.45
```

则进入 `finalize_action`。

否则进入 `rule_fallback`。

### 4. finalize_action

该节点会规范化 DeepSeek 输出，保证字段稳定：

```python
{
    "action": "...",
    "kb_id": "...",
    "text": "...",
    "mentions": [...],
    "mention_aliases": {...},
    "run_requested": True,
    "reply": None,
    "confidence": 0.95
}
```

### 5. rule_fallback

如果没有 DeepSeek 或 DeepSeek 低置信，系统使用本地规则兜底。

规则可以识别：

- `帮助`
- `清空`
- `换成CCKS知识库`
- `文本：...`
- `实体是 ...`
- `运行`

这样保证离线环境也能演示。

## 五、对话状态到链接请求

`ConversationalAgent._apply_action()` 会把 action 写入 `DialogueState`。

如果 DeepSeek 没有返回 `kb_id`，系统会调用 `infer_kb_id()` 做启发式知识库路由。

例如文本中包含：

```text
电影
导演
演员
小说
作品
《》
```

且当前还是 `sample-energy-v1`，则自动切到：

```text
ccks2019-v1
```

如果文本中包含：

```text
电网
国网
南方电网
配电
能源
风电
光伏
```

则保留示例能源知识库。

## 六、Mention 构造与别名扩展

`build_mentions()` 会把 mention 文本转换成 `MentionRecord`。

例子：

```text
文本：李导演的《断背山》真是令人动人
mentions：李导演, 断背山
mention_aliases：{"李导演": ["李安"]}
```

生成：

```python
[
    MentionRecord(
        mention_id="m1",
        text="李导演",
        metadata={"candidate_aliases": ["李安"]}
    ),
    MentionRecord(
        mention_id="m2",
        text="断背山",
        metadata={"candidate_aliases": []}
    )
]
```

这里有一个重要设计：

```text
原始 mention 仍然保留为“李导演”
“李安”只作为候选召回扩展
```

这样既保留用户原文，又允许系统利用上下文代称做更强召回。

## 七、知识库加载与 CCKS 子集筛选

如果使用 `ccks2019-v1`，终端模式会先筛选 CCKS 子知识库。

筛选使用的 alias_texts 包含：

```text
mention_texts + mention_aliases
```

例如：

```text
李导演
断背山
李安
```

这样会从 CCKS2019 的 `kb_data` 中加载：

- 所有别名或 subject 命中 `断背山` 的实体。
- 所有别名或 subject 命中 `李安` 的实体。

这一步可以避免每次在终端演示时加载整个大知识库。

## 八、实体链接工作流

实体链接工作流定义在 `EntityLinkingWorkflow`。

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

## 九、validate_input 节点

该节点负责入口校验。

检查内容：

- `text` 是否为空。
- `mentions` 是否为空。
- `mention_id` 是否为空。
- mention 文本是否为空。
- `start` / `end` 是否非法。
- `top_k_candidates` 是否大于 0。
- `nil_threshold` 是否在 0 到 1。
- `ambiguity_margin` 是否在 0 到 1。

如果失败：

```text
route_decision = invalid_input
validation_errors = [...]
```

流程直接进入 `build_response`，不会继续加载知识库。

## 十、load_kb 节点

该节点加载知识库。

知识库来源有三种：

| 来源 | 场景 |
| --- | --- |
| `sample-energy-v1` | 默认示例知识库 |
| `ccks2019-v1` | CCKS2019 大知识库 |
| `inline_entities` | 终端筛选出的 CCKS 子集或 API 直接传入的实体 |

终端 CCKS 演示通常走的是：

```text
knowledge_base_id = ccks2019-inline
inline_entities = 筛选后的实体列表
```

## 十一、generate_candidates 节点

该节点对每个 mention 生成候选实体。

候选召回逻辑：

```text
MentionRecord
  -> query_texts
       原始 mention
       candidate_aliases
  -> alias_index 精确匹配
  -> fuzzy alias scan
  -> CandidateScore 列表
```

例如 `李导演`：

```text
query_texts = ["李导演", "李安"]
```

如果候选通过 `李安` 命中，则候选原因中会出现：

```text
llm_alias_expansion
```

这表示候选来自 LLM 生成的别名扩展。

## 十二、candidate_route 节点

该节点判断候选是否为空。

如果所有 mention 都没有候选：

```text
candidate_route = empty_candidates
-> nil_fallback
```

如果至少一个 mention 有候选：

```text
candidate_route = has_candidates
-> rerank_candidates
```

注意：如果只有部分 mention 没有候选，流程仍进入 `rerank_candidates`，后续没有候选的 mention 会在 `resolve_mentions` 阶段单独变成 NIL。

## 十三、nil_fallback 节点

如果所有 mention 都没有候选，系统不会强行链接。

输出：

```text
status = nil
confidence = 0.0
linked_entity_id = None
canonical_name = None
rationale = ["candidate_route_empty", "nil_fallback"]
```

这用于处理知识库不存在的实体，例如 2019 年 CCKS 知识库中没有 DeepSeek 公司。

## 十四、rerank_candidates 节点

该节点对召回候选重打分。

重打分公式：

```text
final_score =
  0.62 * alias_similarity
  + 0.23 * context_score
  + type_bonus
  + canonical_bonus
  + expansion_bonus
  + prior_bonus
```

各项含义：

| 分数项 | 含义 |
| --- | --- |
| `alias_similarity` | mention 或扩展别名与知识库别名的相似度 |
| `context_score` | 上下文关键词和实体描述是否匹配 |
| `type_bonus` | mention 类型和实体类型是否一致 |
| `canonical_bonus` | mention 是否直接等于标准名 |
| `expansion_bonus` | 候选是否来自 LLM 别名扩展 |
| `prior_bonus` | CCKS 训练集别名先验支持 |

`llm_alias_expansion` 的加权是小权重，不是让大模型直接决定实体。

当前逻辑：

```python
expansion_bonus = 0.08 if "llm_alias_expansion" in candidate.reasons else 0.0
```

它的作用是：

```text
DeepSeek 提供候选提示
链接器根据上下文、别名、先验继续重排
最终实体 ID 仍必须来自知识库
```

## 十五、resolve_mentions 节点

该节点把候选列表变成最终链接决策。

决策逻辑：

```text
无候选
  -> nil

top1 分数 < nil_threshold
  -> nil

top1 与 top2 分差 < ambiguity_margin
  -> ambiguous + needs_review

否则
  -> linked
```

这一步就是实体消歧的核心。

例如多个 `杭州`：

```text
杭州 164167
杭州 100272
杭州 206331
...
```

系统会根据上下文，比如“企业坐落于杭州”，优先选择地点/城市义项。

## 十六、共指处理

共指在 `EntityLinker._try_coreference()` 中完成。

当前支持的共指词：

```text
该公司
该企业
该机构
该集团
其
他
她
它
```

也支持显式提示：

```python
metadata={"coreference_hint": True}
```

例子：

```text
国网智能科技正在建设算法平台。该公司强调数据治理能力。
```

处理过程：

```text
m_org: 国网智能科技
  -> linked 到 org:sgit

m_coref: 该公司
  -> 没有直接候选
  -> 触发 coreference_fallback
  -> 继承最近同类型实体 m_org
```

输出：

```text
coreference_source_mention_id = m_org
rationale = ["coreference_fallback", "inherits_recent_link:m_org"]
```

终端会打印：

```text
共指来源: m_org
```

## 十七、review_route 节点

该节点判断是否需要人工复核。

进入复核的情况：

- `status = ambiguous`
- `needs_review = True`
- 有候选但置信度低于阈值
- 共指回链结果

如果需要复核：

```text
review_route = needs_review
-> human_review
```

否则：

```text
review_route = auto_accept
-> build_response
```

## 十八、human_review 节点

当前版本没有真正接人工审核系统，而是标记结果。

它会在 evidence 中加入：

```text
human_review_required
```

并在节点事件中记录：

```json
{
  "node": "human_review",
  "detail": {
    "review_required": 1,
    "items": [
      {
        "mention_id": "m_coref",
        "status": "linked",
        "confidence": 0.75
      }
    ],
    "policy": "low_confidence_or_ambiguous"
  }
}
```

## 十九、build_response 节点

该节点构造最终响应。

响应核心字段：

```json
{
  "trace_id": "t10-...",
  "kb_id": "ccks2019-inline",
  "kb_version": "inline",
  "workflow_engine": "langgraph",
  "graph_nodes": [
    "validate_input",
    "load_kb",
    "generate_candidates",
    "rerank_candidates",
    "resolve_mentions",
    "build_response",
    "persist_trace"
  ],
  "route_decision": "resolved",
  "validation_errors": [],
  "results": [...],
  "summary": {...},
  "decision_log": [...]
}
```

`summary` 示例：

```json
{
  "total_mentions": 2,
  "linked": 2,
  "ambiguous": 0,
  "nil": 0,
  "review_required": 0
}
```

## 二十、persist_trace 节点

该节点保存完整执行结果。

默认写入：

```text
artifacts/traces/
```

如果目录权限受限，会回退到系统临时目录：

```text
%TEMP%/topic10_entity_linking_agent/traces
```

如果仍失败，会保留进程内缓存。

Trace 用途：

- 审计每次请求。
- 回放 badcase。
- 查看 LangGraph 节点轨迹。
- 对比候选、分数、证据和最终决策。

## 二十一、终端输出如何阅读

终端输出分为三块：

### 1. 运行摘要

```text
trace_id: t10-...
workflow_engine: langgraph
route_decision: resolved
graph_nodes: validate_input -> load_kb -> generate_candidates -> ...
统计: mention=2, linked=2, ambiguous=0, nil=0, review=0
```

含义：

| 字段 | 含义 |
| --- | --- |
| `trace_id` | 本次执行唯一追踪 ID |
| `route_decision` | 本次最终路由 |
| `graph_nodes` | 实际走过的 LangGraph 节点 |
| `统计` | linked / nil / ambiguous / review 汇总 |

### 2. 链接结果

```text
[m1] 李导演
  状态: linked
  置信度: 0.872
  标准实体: 李安
  实体ID: 349056
  依据: llm_alias_expansion / description_overlap_support / score_above_link_threshold
  Top候选:
    - 李安 (349056) score=0.872
    - 李安 (165842) score=0.777
```

这里可以看到：

- 最终实体。
- 置信度。
- 消歧依据。
- top 候选列表。

### 3. 节点事件

```text
- validate_input: {'valid': True, 'errors': []}
- load_kb: {'kb_id': 'ccks2019-inline', 'entity_count': 47}
- generate_candidates: {'mention_count': 2, 'candidate_count': 47}
- resolve_mentions: {'linked': 2, 'nil': 0, 'ambiguous': 0}
```

节点事件适合答辩时说明“Agent 不是黑盒，而是可追踪工作流”。

## 二十二、样例一：李导演与断背山

输入：

```text
李导演的《断背山》真是令人动人 其中实体是李导演和断背山
```

对话理解结果：

```json
{
  "kb_id": "ccks2019-v1",
  "text": "李导演的《断背山》真是令人动人",
  "mentions": ["李导演", "断背山"],
  "mention_aliases": {
    "李导演": ["李安"]
  }
}
```

实体链接结果：

```text
李导演 -> 李安 -> 349056
断背山 -> 断背山 -> 83393
```

关键点：

- `李导演` 不是知识库标准别名。
- DeepSeek 根据上下文扩展出 `李安`。
- 候选召回使用 `李导演` 和 `李安` 两个查询。
- 重排器根据 `断背山` 上下文把华人导演李安排到 top1。
- 最终 ID 仍来自 CCKS 知识库，不由 DeepSeek 编造。

## 二十三、样例二：DeepSeek 公司 NIL

输入：

```text
deepseek这家互联网企业坐落于杭州 实体为deepseek 互联网企业 杭州
```

可能结果：

```text
deepseek -> nil
互联网企业 -> nil
杭州 -> 杭州(城市义项)
```

原因：

- CCKS2019 是 2019 年数据，通常没有 DeepSeek 公司。
- `互联网企业` 是类别词，不是具体实体。
- `杭州` 在知识库中有多个义项，系统根据上下文选择地点义项。

这说明系统不会强行链接知识库外实体。

## 二十四、样例三：共指回链

输入：

```text
国网智能科技正在建设算法平台。该公司强调数据治理能力。
```

mention：

```text
国网智能科技
该公司
```

结果：

```text
国网智能科技 -> 国网智能科技股份有限公司 -> org:sgit
该公司 -> 国网智能科技股份有限公司 -> org:sgit
共指来源: m_org
```

数据流：

```text
第一个 mention linked
第二个 mention 触发 coreference_fallback
继承最近同类型实体
标记 needs_review
进入 human_review
```

## 二十五、API 模式数据流

API 请求入口：

```text
POST /v1/link
```

请求体示例：

```json
{
  "text": "国家电网与南方电网联合发布标准。",
  "mentions": [
    {
      "mention_id": "m1",
      "text": "国家电网",
      "entity_type": "organization"
    }
  ],
  "knowledge_base_id": "sample-energy-v1"
}
```

数据流：

```text
LinkRequestView
  -> to_service_kwargs()
  -> Topic10EntityLinkingService.link()
  -> EntityLinkingWorkflow
  -> LinkResponseView
```

API 模式默认不调用 DeepSeek，因为调用方已经提供了结构化字段。

## 二十六、各类异常或复杂情况的处理

| 情况 | 处理方式 |
| --- | --- |
| 输入为空 | `validate_input` 短路，返回 `invalid_input` |
| 所有 mention 无候选 | `candidate_route -> nil_fallback` |
| 单个 mention 无候选 | `resolve_mentions` 中输出 `nil` |
| top1 分数过低 | 输出 `nil` 或进入复核 |
| top1 / top2 分差过小 | 输出 `ambiguous`，进入 `human_review` |
| 共指词出现 | 尝试继承最近同类型实体 |
| 知识库没有实体 | 输出 `nil`，不编造实体 ID |
| DeepSeek 失败 | 进入 `rule_fallback` |
| DeepSeek 给了别名扩展 | 作为候选召回和重排信号 |

## 二十七、为什么说它是 Agent 工作流

它不是单纯函数调用，而是有状态、有分支、有工具调用、有追踪的工作流。

体现为：

- 对话层维护 `DialogueState`。
- DeepSeek 只是一个 `llm_understand` 节点。
- 低置信会自动 fallback。
- 实体链接层有 `candidate_route` 和 `review_route` 条件边。
- NIL、ambiguous、human_review 都是显式分支。
- 每次执行都有 `trace_id` 和 `node_events`。

所以整个系统具备：

- 多轮对话能力。
- 知识库路由能力。
- 候选扩展能力。
- 消歧能力。
- NIL 检测能力。
- 共指回链能力。
- 人工复核标记能力。
- 可追溯审计能力。

## 二十八、一句话总结

本项目的数据流可以概括为：

```text
用户输入被对话工作流解析为结构化 action，
action 被转换为 mention 与知识库请求，
实体链接工作流通过校验、候选、重排、消歧、NIL、共指和复核节点生成最终结果，
所有节点轨迹和证据都会写入响应与 trace。
```

