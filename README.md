# Entity Link Agent

实体链接与知识对齐智能体（课题 10），对外提供归一化的 HTTP API 服务。将文本中已识别的实体指称（mention）链接到知识库中的标准实体，输出标准实体 ID、置信度和可追溯依据。

## 快速启动

```powershell
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .

# 2. 配置环境变量
cp .env.example .env

# 3. 下载嵌入模型（首次需要，约 184MB）
.\.venv\Scripts\python.exe scripts/download_model.py


# 4. 启动服务
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后：
- 前端界面：http://localhost:8000/
- 接口文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## Docker 部署

```bash
cp .env.example .env
docker-compose up
```

## 环境配置

### `.env` — 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APP_HOST` | `0.0.0.0` | 服务监听地址 |
| `APP_PORT` | `8000` | 服务端口 |
| `APP_LOG_LEVEL` | `info` | 日志级别 |
| `KB_DIR` | `data/knowledge_bases` | 知识库存储目录 |

### `config.yaml` — 业务配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `linker.default_top_k` | 5 | 候选召回数量 |
| `linker.nil_threshold` | 0.6 | NIL 判定阈值（仅对相似度命中生效） |
| `linker.enable_nil` | true | 是否启用 NIL 检测 |
| `linker.enable_coreference` | true | 是否启用共指消解 |
| `coreference.trigger_terms` | 该公司、其、他... | 共指触发词列表 |
| `embedding.model_name` | BAAI/bge-small-zh-v1.5 | 嵌入模型路径 |
| `embedding.device` | auto | 推理设备（auto 自动检测 / cuda / cpu） |
| `embedding.index_dir` | data/vector_index | FAISS 索引存储路径 |

## 项目依赖

```
fastapi>=0.115        # Web 框架
pydantic>=2.7         # 数据校验
langgraph>=0.2        # 流水线编排
sentence-transformers >=3.0  # 文本嵌入
faiss-cpu>=1.8        # 向量检索
PyYAML>=6.0           # 配置解析
uvicorn>=0.30         # ASGI 服务器
```

完整依赖见 `pyproject.toml`。

## 目录结构

```
entity_link_agent/
├── app/
│   ├── main.py                   # FastAPI 入口，日志配置，静态文件挂载
│   ├── config.py                 # 配置加载（.env + config.yaml → AppConfig）
│   ├── dependencies.py           # 依赖注入容器（单例缓存）
│   ├── api/
│   │   ├── router.py             # 路由注册
│   │   └── v1/
│   │       ├── health.py         # GET /health
│   │       ├── knowledge_bases.py# POST/GET 知识库导入与管理
│   │       └── entity_link.py    # POST /api/v1/entity-link
│   ├── models/                   # Pydantic 数据模型
│   │   ├── enums.py              # LinkStatus / EvidenceType / EntityType
│   │   ├── entity.py             # Entity / KnowledgeBase / KBPackage
│   │   ├── request.py            # LinkRequest / LinkOptions / MentionInput
│   │   └── response.py           # LinkResponse / LinkResult / CoreferenceChain
│   ├── services/                 # 服务层
│   │   ├── kb_service.py         # 知识库管理业务逻辑
│   │   └── link_service.py       # 实体链接主流程（LangGraph 编排）
│   ├── core/                     # 核心能力层（每文件一个能力，互相独立）
│   │   ├── candidate.py          # 候选召回（精确名称 + 向量语义）
│   │   ├── scorer.py             # 分档评分（按 match_source 选权重）
│   │   ├── nil_detector.py       # NIL 判定（精确命中豁免）
│   │   ├── coreference.py        # 共指消解（触发词前置处理）
│   │   ├── evidence.py           # 链接依据生成
│   │   └── embedder.py           # 文本向量化（SentenceTransformer）
│   ├── storage/                  # 存储层
│   │   ├── kb_store.py           # JSON 文件读写 + 向量索引管理
│   │   ├── index.py              # 名称索引（归一化字符串 → 实体）
│   │   └── vector_index.py       # FAISS 索引封装
│   └── middleware/               # 中间件
│       ├── logging.py            # 请求日志
│       └── error_handler.py      # 统一异常处理
├── static/
│   └── index.html                # 前端界面
├── tests/
│   ├── conftest.py               # 测试 fixture
│   ├── test_api.py               # 集成测试
│   └── fixtures/                 # 测试用 JSON 数据
│       ├── cn_enterprises_kb.json # 120 实体中国企业库
│       ├── link_alias.json        # 别名命中测试
│       ├── link_canonical.json    # 标准名命中测试
│       ├── link_coref.json        # 共指消解测试
│       ├── link_nil.json          # NIL 检测测试
│       ├── link_disambig.json     # 消歧测试
│       ├── link_former_name.json  # 曾用名匹配测试
│       ├── link_full.json         # 综合多类型测试
│       ├── link_fuzzy.json        # 语义召回阈值样例（沿用旧文件名）
│       ├── link_threshold.json    # 阈值边界测试
│       ├── link_semantic.json     # 纯语义匹配测试
│       ├── link_context_disambig.json
│       ├── link_multi_sector.json
│       └── link_nested.json
├── scripts/
│   └── download_model.py         # 嵌入模型下载脚本
├── doc/                          # 设计文档
├── config.yaml                   # 业务逻辑配置
├── .env.example                  # 环境变量模板
├── pyproject.toml                # 项目元信息与依赖
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/v1/knowledge-bases` | 导入知识库（含实体，同名覆盖） |
| `POST` | `/api/v1/knowledge-bases/import-file` | 从服务器本地文件转换并导入知识库 |
| `POST` | `/api/v1/knowledge-bases/import-upload` | 从浏览器选择并上传文件转换为知识库 |
| `GET` | `/api/v1/knowledge-bases` | 列出所有知识库 |
| `GET` | `/api/v1/knowledge-bases/{kb_id}` | 查看知识库详情 |
| `POST` | `/api/v1/entity-link` | 实体链接（核心接口） |

> 注：`POST /api/v1/knowledge-bases/{kb_id}/entities` 已移除，统一为一步导入。
> 文件导入支持 KBPackage JSON、实体数组 JSON/JSONL、CCKS `kb_data`、PDF 和文本；PDF/文本需要配置大模型 API 才能稳定抽取实体。网页上的“选择文件”使用 `/import-upload`，会弹出系统文件选择窗口。

## 架构

### 分层架构

```
HTTP 请求
  ↓ 中间件层（logging、error_handler）
  ↓ 路由层（api/v1/）
  ↓ 服务层（services/）
  ↓ 核心能力层（core/）
  ↓ 存储层（storage/）
```

每层只向下调用，不向上。各核心模块独立可测。

### 实体链接流水线

```
POST /api/v1/entity-link
        │
┌───────▼──────────┐
│ 1. validate      │  校验请求参数、KB 存在性、版本匹配
└───────┬──────────┘
        │
┌───────▼──────────┐
│ 2. load_kb       │  加载 NameIndex + VectorIndex
└───────┬──────────┘
        │
┌───────▼──────────┐
│ 3. candidates    │  候选召回（精确名称 → 向量语义）
│                  │  触发词跳过候选，走共指消解
└───────┬──────────┘
        │
┌───────▼──────────┐
│ 4. rerank        │  按 match_source 分档评分：
│                  │    canonical    → 0.85·向量 + 0.15·上下文
│                  │    alias        → 0.82·向量 + 0.18·上下文
│                  │    former_name  → 0.80·向量 + 0.20·上下文
│                  │    similarity   → 0.55·向量 + 0.45·上下文
└───────┬──────────┘
        │
┌───────▼──────────┐
│ 5. resolve       │  NIL 判定：精确命中（canonical/alias/former）豁免
│                  │  相似度命中按 nil_threshold 阈值判定
└───────┬──────────┘
        │
┌───────▼──────────┐
│ 6. coreference   │  共指消解：触发词回指前文实体，构建共指链
└───────┬──────────┘
        │
┌───────▼──────────┐
│ 7. response      │  组装 LinkResponse（含依据、候选、共指链）
└──────────────────┘
```

### 候选召回：多源合并（非二选一）

```
retrieve(mention)
    ├── ① NameIndex.lookup()  → 精确命中 → score ≥ 0.88
    ├── ② FAISS.search()      → 向量相似 → score 0.5~0.85
    └── ③ BGE 向量检索         → 语义召回 → cosine score
              │
              ▼
         合并去重，精确命中优先
```

### 评分策略

| 命中方式 | 向量权重 | 上下文权重 | NIL 豁免 | 保底分数 |
|---------|:------:|:-------:|:------:|:------:|
| canonical_match | 0.85 | 0.15 | ✅ | 0.95 |
| alias_match | 0.82 | 0.18 | ✅ | 0.92 |
| former_name_match | 0.80 | 0.20 | ✅ | 0.88 |
| similarity_match | 0.55 | 0.45 | ❌ | — |

### 关键设计决策

1. **共指前置** — 触发词在候选召回前拦截，不走向量检索，避免误链
2. **精确命中不判 NIL** — canonical/alias/former_name 直接 linked，不过阈值
3. **缺失文件容错** — `ccks2019_alias_prior.json` 不存在时先验概率为 0，不影响运行
4. **向量索引按需构建** — 导入时失效旧索引，首次链接时自动重建

## 使用示例

### 1. 导入知识库（一步完成）

```powershell
$body = Get-Content tests/fixtures/cn_enterprises_kb.json -Raw -Encoding UTF8
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/knowledge-bases" `
  -ContentType "application/json" -Body $body
```

响应：
```json
{
  "kb_id": "cn-enterprises-v1",
  "kb_version": "v1",
  "entity_count": 120,
  "status": "created"
}
```

> 再次导入同名 kb_id 会覆盖，返回 `"status": "overwritten"`。

### 2. 实体链接

```powershell
$body = Get-Content tests/fixtures/link_full.json -Raw -Encoding UTF8
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/entity-link" `
  -ContentType "application/json" -Body $body
```

### 3. 从本地文件导入知识库

```powershell
$body = @{
  file_path = "E:\平时\2026实训\阶段2\ccks2019_el\kb_data"
  kb_id = "ccks2019-v1"
  kb_version = "v1"
  description = "CCKS2019 实体链接知识库"
  source_type = "ccks_kb_data"
  import_to_store = $true
  include_entities = $false
  preview_limit = 5
  use_llm = $false
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/knowledge-bases/import-file" `
  -ContentType "application/json" -Body $body
```

网页模式可以直接打开 `http://localhost:8000/`，在“知识库管理 -> 从本地文件导入”中点击“选择文件”，选择 `.json`、`.jsonl`、`.pdf`、`.txt` 或 `.md` 文件后上传。

如果导入 PDF 或纯文本，并希望自动抽取实体，需要先在 `.env` 配置：

```powershell
DEEPSEEK_API_KEY=你的key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

### 3. 调节 NIL 阈值测试

```powershell
# 严格模式 — 只保留高置信度链接
$body = (Get-Content tests/fixtures/link_fuzzy.json -Raw -Encoding UTF8) `
  -replace '"nil_threshold": 0.6', '"nil_threshold": 0.80'
Invoke-RestMethod ... -Body $body

# 宽松模式 — 更多提及被链接
$body = (Get-Content tests/fixtures/link_semantic.json -Raw -Encoding UTF8) `
  -replace '"nil_threshold": 0.6', '"nil_threshold": 0.45'
Invoke-RestMethod ... -Body $body
```

### 4. 前端界面

浏览器打开 `http://localhost:8000/`：

- **知识库管理**：点「填入示例」→ 点「导入知识库」
- **实体链接**：点「填入示例」→ 点「执行链接」
- 支持直接粘贴完整 LinkRequest JSON

## 测试

```powershell
# 运行全部测试
.\.venv\Scripts\python.exe -m pytest tests/ -v

# 运行单个测试
.\.venv\Scripts\python.exe -m pytest tests/test_api.py::test_entity_link_coreference -v
```

## 团队开发指引

### 首次克隆后

```powershell
git clone <repo-url>
cd entity_link_agent
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
cp .env.example .env
.\.venv\Scripts\python.exe scripts/download_model.py
```

### 开发流程

1. 从 `main` 分支拉最新代码
2. 创建功能分支：`git checkout -b feat/xxx`
3. 开发并确保 `pytest tests/` 全绿
4. 提交并推送，发起 PR

### 添加新测试

1. KB 数据放 `tests/fixtures/` 目录（JSON 格式）
2. 链接请求数据放 `tests/fixtures/link_*.json`
3. 测试用例添加到 `tests/test_api.py`
4. 需覆盖的典型场景：
   - 别名命中（`link_alias.json`）
   - 共指消解（`link_coref.json`）
   - NIL 检测（`link_nil.json`）
   - 消歧场景（`link_disambig.json`）
   - 综合场景（`link_full.json`）
   - 阈值边界（`link_threshold.json`）
   - 语义匹配（`link_semantic.json`）

### 配置变更

- 业务逻辑配置修改 `config.yaml`（提交仓库）
- 环境变量修改 `.env`（不提交，参考 `.env.example`）
- 新增依赖修改 `pyproject.toml`（提交仓库）

### 注意事项

- **GPU / CPU**：`config.yaml` 中 `device: auto` 会自动检测 CUDA 可用性。有 NVIDIA 显卡 → 自动用 GPU；无显卡 → CPU，速度较慢但可运行
- **PyTorch 依赖**：由 `sentence-transformers` 自动安装，无需手动声明。如需 GPU 加速，确保 PyTorch 为 CUDA 版本（`pip show torch | grep cuda`）
- **FAISS**：使用 `faiss-cpu`（PyPI 预编译包）。GPU 版 `faiss-gpu` 需额外配置，当前方案已足够处理万级实体
- 数据目录 `data/knowledge_bases/`、`data/vector_index/`、`data/models/` 已加入 `.gitignore`，不提交
- 嵌入模型需每位开发者本地下载：`python scripts/download_model.py`
- 首次实体链接请求会加载 184MB BGE 模型，耗时约 5 秒（GPU）到 15 秒（CPU），后续请求秒级响应
- 前端直接打开 `static/index.html` 会走 `file://` 协议，自动指向 `localhost:8000`
