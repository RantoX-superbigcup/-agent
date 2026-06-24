# Entity Link Agent

实体链接与知识对齐智能体，对外提供归一化的 HTTP API 服务。

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
cp .env.example .env
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

服务启动后：
- 接口文档：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

## Docker 部署

```bash
cp .env.example .env
docker-compose up
```

## 目录结构

```
entity_link_agent/
├── app/
│   ├── main.py                   # FastAPI 入口
│   ├── config.py                 # 配置加载
│   ├── dependencies.py           # 依赖注入
│   ├── api/v1/                   # 路由层
│   │   ├── health.py
│   │   ├── knowledge_bases.py
│   │   └── entity_link.py
│   ├── models/                   # Pydantic 数据模型
│   ├── services/                 # 服务层（LangGraph 工作流）
│   ├── core/                     # 核心能力层
│   │   ├── candidate.py          # 候选实体召回
│   │   ├── scorer.py             # 候选打分
│   │   ├── nil_detector.py       # NIL 检测
│   │   ├── coreference.py        # 共指消解
│   │   └── evidence.py           # 链接依据生成
│   ├── storage/                  # 存储层
│   └── middleware/               # 日志、错误处理
├── data/
│   ├── knowledge_bases/          # 知识库 JSON 文件
│   └── models/                   # alias prior 模型
├── tests/
├── config.yaml                   # 业务逻辑配置
├── .env.example                  # 环境变量模板
├── Dockerfile
└── docker-compose.yml
```

## 接口一览

```
GET  /health
POST /api/v1/knowledge-bases
POST /api/v1/knowledge-bases/{kb_id}/entities
GET  /api/v1/knowledge-bases
GET  /api/v1/knowledge-bases/{kb_id}
POST /api/v1/entity-link
```

## 使用示例

**1. 创建知识库**

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/knowledge-bases" `
  -ContentType "application/json" `
  -Body '{"kb_id":"kb-001","kb_version":"v1","description":"能源行业实体库"}'
```

**2. 导入实体**

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/knowledge-bases/kb-001/entities" `
  -ContentType "application/json" `
  -Body '{"entities":[{"entity_id":"E001","canonical_name":"国网江苏省电力有限公司","entity_type":"ORG","aliases":["国网江苏电力","江苏电力"],"former_names":[],"description":"国家电网在江苏的省级电力公司","keywords":["江苏","南京","电力"]}]}'
```

**3. 实体链接**

```powershell
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/api/v1/entity-link" `
  -ContentType "application/json" `
  -Body '{
    "schema_version":"v1",
    "request_id":"req-001",
    "text":{"content":"国网江苏电力完成南京供电保障。该公司表示将继续提升可靠性。"},
    "mentions":[
      {"mention_id":"m1","surface_form":"国网江苏电力","start_offset":0,"end_offset":6},
      {"mention_id":"m2","surface_form":"该公司","start_offset":19,"end_offset":22}
    ],
    "knowledge_base":{"kb_id":"kb-001","kb_version":"v1"}
  }'
```

## 运行测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

## 配置说明

`config.yaml` 控制链接参数默认值、共指触发词等业务逻辑配置。

`.env` 控制端口、路径、日志级别等环境配置，参照 `.env.example` 创建。
