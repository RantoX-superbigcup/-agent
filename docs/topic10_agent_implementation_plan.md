# 课题10 实体链接与知识对齐智能体实现规划

编写目的：根据课题 10 的题目要求，规划实体链接与知识对齐智能体的实现路线、技术方案、工作流设计、数据接入方式和阶段安排。

适用范围：本规划用于后续工程实现、答辩说明、任务书映射和项目迭代。本次仅输出本地文档，不上传 Git。

## 一、课题目标与需求理解

本课题面向“实体链接与知识对齐”任务，目标是在已有实体识别结果的基础上，将文本中的实体 mention 链接到知识库中的标准实体，并在知识库不存在对应实体时输出 NIL。系统不仅要给出最终实体 ID，还要提供候选、置信度、消歧依据、共指来源和执行轨迹，满足可解释、可评测、可集成的智能体工程要求。

从任务边界看，本课题不把命名实体识别作为核心算法目标，而是假设输入中已经给出了待链接 mention。为了增强演示和交互能力，可以在终端 Agent 中接入大模型辅助抽取 mention，但核心服务仍以结构化 mention 输入为准，保证接口稳定。

最终系统应支持三类典型结果：第一，mention 能够链接到知识库标准实体，输出 linked；第二，知识库中没有合适实体，输出 nil；第三，多个候选难以区分或置信度不足，输出 ambiguous 并进入人工复核分支。

## 二、智能体总体定位

本项目规划实现的是一个“可对话、可调用、可追踪”的实体链接智能体，而不是单一函数式匹配程序。智能体需要具备多轮对话状态维护、知识库选择、候选召回、上下文消歧、NIL 检测、共指回链、人工复核标记和 trace 审计能力。

对外形态分为两种：一种是 FastAPI 服务，用于平台或其他系统通过 HTTP API 调用；另一种是终端 CLI Agent，用于课堂展示、答辩演示和交互式调试。二者共享同一套实体链接核心流程，避免出现接口结果和终端结果不一致的问题。

大模型在系统中的定位是“理解与辅助”，不是“直接裁决”。DeepSeek API 可以用于解析自然语言、判断用户意图、补全 mention、选择知识库和辅助判断 mention_type，但最终实体 ID 必须从知识库候选中产生，避免大模型幻觉导致错误链接。

## 三、总体技术方案

系统采用 Python 作为主要开发语言，FastAPI 作为服务化接口框架，LangChain + LangGraph 作为智能体工作流编排框架，CCKS2019-EL 作为主要实验数据集，DeepSeek API 作为可选的大模型理解模块。

FastAPI 负责提供健康检查、知识库管理、实体导入和实体链接接口。LangGraph 负责把实体链接过程拆解成多个可观测节点，并通过条件边处理输入非法、候选为空、低置信、歧义和复核等复杂情况。LangChain 的 RunnableLambda 用于封装每个节点，使后续可以将规则模块替换为检索器、reranker 或 LLM chain。

工程实现上建议保留当前 app/ 服务化结构，同时在后续合并已有的对话 Agent、CCKS2019 适配、DeepSeek 客户端、trace 文档和评测脚本，使最终项目同时满足“接口规范”和“智能体能力”两类要求。

## 四、系统架构设计

系统可以划分为五层：接口层、对话理解层、实体链接工作流层、知识库与索引层、评测与追踪层。接口层负责 HTTP API 和终端输入输出；对话理解层负责将自然语言转换为结构化 action；实体链接工作流层负责核心决策；知识库与索引层负责实体加载、别名索引和候选召回；评测与追踪层负责保存执行过程、计算指标和分析 badcase。

接口层主要包含 /health、/api/v1/knowledge-bases、/api/v1/knowledge-bases/{kb_id}/entities、/api/v1/entity-link 等接口。终端模式下，用户可以直接输入“李导演的《断背山》真是令人动人，其中实体是李导演和断背山”，系统自动解析文本、mention 和知识库，并调用同一个实体链接服务。

知识库层需要同时支持示例知识库和 CCKS2019 知识库。示例知识库用于快速演示和单元测试，CCKS2019 用于正式实验和评测。对于大规模知识库，不能每次全量扫描，应分别建立标准名、别名和曾用名索引，并配合向量索引支撑召回。

## 五、LangGraph 工作流规划

实体链接流程使用 LangGraph 表达为有向图，核心节点为 validate_input、load_kb、generate_candidates、candidate_route、nil_fallback、rerank_candidates、resolve_mentions、review_route、human_review、build_response、persist_trace。每个节点都有明确输入输出，便于调试和替换。

validate_input 节点负责检查文本、mention、知识库、top_k、nil_threshold、ambiguity_margin 等字段是否合法。如果输入为空或 mention 缺失，流程直接短路到 build_response，返回 invalid_input，而不是继续执行无意义的候选召回。

candidate_route 节点负责处理候选为空的复杂情况。如果所有 mention 都没有候选，则进入 nil_fallback，所有结果输出 NIL；如果只有部分 mention 没有候选，则继续进入 rerank_candidates，后续在 resolve_mentions 中对单个 mention 分别判定。

review_route 节点负责处理低置信和歧义情况。当 top1 分数低于阈值、top1 与 top2 分差小于 ambiguity_margin、或者结果来自共指回链时，进入 human_review 分支。当前阶段可以先标记 human_review_required，后续再对接真实人工审核台。

## 六、对话 Agent 规划

对话 Agent 维护 DialogueState，包含当前知识库、当前文本、mention 列表以及是否已经准备运行。用户可以多轮输入，例如先切换知识库，再输入文本，再补充 mention，最后输入“运行”触发实体链接。

对话理解工作流包含 llm_understand、route_after_llm、finalize_action、rule_fallback 四个节点。配置 DeepSeek API 时优先使用大模型解析用户意图；未配置或低置信时，使用本地规则兜底，支持“帮助”“清空”“换成 CCKS 知识库”“文本：...”“实体是 ...”“运行”等命令。

大模型在对话层只负责判断用户意图、抽取标准 LinkRequest 所需的 text 和 mentions，以及在需要时辅助判断 mention_type。候选实体必须由知识库召回层产生，不能由大模型直接扩展出额外候选名称并绕过知识库检索。

## 七、CCKS2019 数据接入方案

CCKS2019-EL 数据集包含知识库、训练集和测试集。知识库中的每个实体需要转换为统一的 KnowledgeBaseEntity，包括 entity_id、canonical_name、aliases、entity_type、description、keywords 和 metadata。训练集中的 mention_data 可转换为 MentionRecord，用于评测和统计别名先验。

候选生成阶段应优先使用名称索引，而不是逐实体扫描。索引字段包括标准名、别名和曾用名。对于“杭州”这类多义实体，候选列表中可能出现多个同名项，因此必须保留候选池并交给重排器做消歧。

训练集可以用于构建 alias prior，即统计某个 mention 更常链接到哪个 entity_id。例如 mention “断背山”在训练集中经常指向电影实体，则该实体获得先验加权。alias prior 只能作为小权重辅助，不能覆盖上下文证据。

## 八、候选召回与消歧方案

候选召回采用三段式策略。第一步是严格精确匹配，只在 mention.surface_form 与标准名、别名或曾用名完全相等时命中。第二步是在精确匹配失败后直接进入模糊召回，对每个实体对象的标准名、别名、曾用名分别计算字符串相似度，取该实体最高分作为模糊召回分数。第三步是在模糊候选数量不足时，使用 BGE 向量检索补足候选池。当前召回层主输出已经调整为轻量引用结构，即按 mention 产出候选 `entity_id`、`recall_source`、`match_slot` 和 `recall_status`，随后再由路由节点判断是直接链接、进入消歧、等待共指，还是进入 NIL 路径。`CandidateResult` 目前仍作为后半段兼容对象保留，用于衔接旧的重排、NIL 决策和证据生成逻辑，但不再是召回层的主输出格式。

候选重排采用可解释加权公式，综合 alias_similarity、context_score、description_overlap、entity_type_bonus、canonical_bonus 和 alias_prior_bonus。这样既能处理标准名称命中，也能利用上下文与先验证据完成消歧。

消歧阶段根据 top1 分数、NIL 阈值和 top1/top2 分差做决策。若 top1 分数低于 nil_threshold，则输出 NIL；若 top1 与 top2 分差小于 ambiguity_margin，则输出 ambiguous 并进入复核；否则输出 linked。

## 九、NIL、歧义与共指处理方案

NIL 用于表示知识库中没有对应实体。系统不能为了提高链接数量而强行链接，例如 CCKS2019 是 2019 年数据，通常不包含 DeepSeek 公司，因此“deepseek 这家互联网企业”应输出 NIL，而不是链接到无关实体。

歧义处理用于同名或近似实体较多的场景。例如知识库中可能存在多个“杭州”，系统应利用上下文“坐落于杭州”“杭州城市”“杭州电视剧”等语义线索区分地点、作品、机构或其他实体。如果证据不足，则进入人工复核。

共指处理用于“该公司”“该企业”“其”“他”“她”“它”等代词或指代短语。当前阶段可以采用规则和最近实体回链策略，后续可以接入更复杂的共指消解模型。共指结果应输出 coreference_source_mention_id，并建议进入复核。

## 十、接口与输出规范

HTTP 输入应包含 schema_version、request_id、text、mentions、knowledge_base 和 options。mentions 中至少包含 mention_id、surface_form、start_offset、end_offset。options 中包含 top_k、nil_threshold、enable_nil、enable_coreference、return_candidates、return_evidence 等参数。

HTTP 输出应包含 request_id、status、results、coreference_chains、summary 和 trace。每个 result 包含 mention_id、surface_form、link_status、entity、confidence、candidates、evidence 和 coreference。这样调用方既能拿到最终结果，也能看到候选与证据。

终端输出应包含运行摘要、链接结果和节点事件三部分。运行摘要展示 trace_id、workflow_engine、route_decision 和统计信息；链接结果展示每个 mention 的实体、置信度、依据和 top 候选；节点事件展示 LangGraph 各节点的执行结果。

## 十一、评测方案

评测数据主要来自 CCKS2019-EL 训练集和测试集。训练集可用于构建别名先验和调参，测试集用于最终评估。评价指标包括 Precision、Recall、F1、Accuracy、NIL Accuracy、Ambiguous 命中率以及 Top-K Recall。

除整体指标外，还需要保留 badcase 分析。badcase 应记录原文、mention、gold entity、预测 entity、top 候选、置信度、触发节点和失败原因。常见失败类型包括候选召回失败、同名实体消歧失败、NIL 误判、共指错误和大模型复核误判。

评测脚本应支持批量运行，并输出 JSON 或 CSV 报告。为了答辩展示，可以选择若干典型样例，包括“李导演 -> 李安”“断背山 -> 电影实体”“DeepSeek 公司 -> NIL”“该公司 -> 前文机构共指”等。

## 十二、实施阶段安排

第一阶段完成环境和基础接口。目标是保证 FastAPI 服务可启动，知识库可导入，/api/v1/entity-link 可返回 linked、nil、ambiguous 基础结果。当前 resort_struct 分支可以作为这一阶段基础。

第二阶段补齐 LangGraph 复杂分支。重点实现 validate_input、candidate_route、review_route 和 human_review，使流程具备处理异常输入、候选为空、低置信和歧义复核的能力。

第三阶段接入 CCKS2019。完成数据格式转换、别名索引、候选子集加载、alias prior 统计和批量评测脚本。此阶段需要重点解决同名实体和大知识库召回效率问题。

第四阶段接入终端对话 Agent 和 DeepSeek。实现多轮状态维护、自然语言任务解析、mention_type 辅助判断和规则兜底，保证没有 API Key 时仍可演示基础流程。

第五阶段完善文档和展示。整理架构说明、数据流说明、接口说明、部署说明、评测报告和 badcase 报告，并准备终端演示脚本。

## 十三、风险与应对措施

风险一是知识库实体较多导致候选召回慢。应对方式是建立别名索引和候选子集筛选，避免全量扫描。

风险二是同名实体较多导致误链接。应对方式是引入上下文关键词、实体描述重叠、实体类型和 alias prior 共同打分，并在低置信时进入复核。

风险三是大模型幻觉。应对方式是限制大模型只输出 action 或标准 LinkRequest 所需字段，不允许直接输出最终 entity_id；最终 ID 必须来自知识库候选集合。

风险四是训练集和测试集字段不一致。应对方式是建立独立数据适配层，将外部数据统一转换为内部 Schema。

风险五是答辩时环境不稳定。应对方式是准备 CLI 离线规则兜底、示例知识库和固定演示样例，保证即使没有网络和 DeepSeek API 也可以展示核心流程。

## 十四、预期交付物

预期交付物包括：实体链接 FastAPI 服务、终端对话 Agent、LangGraph 工作流实现、CCKS2019 数据适配脚本、候选召回与重排模块、NIL 与人工复核分支、共指处理模块、trace 持久化模块、评测脚本、接口文档、数据流文档、部署说明和典型演示样例。

最终效果应能够展示：用户输入文本和 mention 后，系统自动选择知识库、生成候选、重排消歧、判断 NIL、处理共指、标记复核，并输出标准实体 ID、实体名称、置信度、候选列表和可解释依据。

一句话概括，本课题的实现路线是：以 LangGraph 为流程骨架，以知识库候选为事实边界，以 DeepSeek 为对话理解辅助，以 CCKS2019 为评测基础，构建一个可服务化、可对话、可解释、可复核的实体链接智能体。
