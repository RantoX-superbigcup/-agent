# 部署说明

## 终端演示

```powershell
cd topic10_entity_linking_agent
.\.venv\Scripts\python -m entity_linking_agent.cli
```

进入后可以直接多轮对话，例如先说“换成CCKS知识库”，再输入短文本和实体列表，最后说“运行”。

CCKS2019 快速演示：

```powershell
.\.venv\Scripts\python -m entity_linking_agent.cli --demo
```

## 本地公开监听

下面的命令会监听 `0.0.0.0:8000`，可被局域网或平台服务通过机器 IP 调用。

```powershell
cd topic10_entity_linking_agent
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python .\main.py
```

如果当前终端是 `(uno)` 并显示 Python 3.9.x，请直接使用这组命令即可；依赖已固定为 Python 3.9 可运行的 LangChain 0.3 / LangGraph 0.2 系列。

等价的 uvicorn 启动方式：

```powershell
.\.venv\Scripts\python -m uvicorn entity_linking_agent.app:app --app-dir src --host 0.0.0.0 --port 8000
```

访问地址：

```text
http://<服务器IP>:8000/health
http://<服务器IP>:8000/docs
http://<服务器IP>:8000/v1/link
```

## 命令行入口

安装后可直接使用 `topic10-agent`：

```powershell
cd topic10_entity_linking_agent
.\.venv\Scripts\python -m pip install -e .
$env:EL_HOST = "0.0.0.0"
$env:EL_PORT = "8000"
.\.venv\Scripts\topic10-agent.exe
```

## Docker 私有化部署

```powershell
cd topic10_entity_linking_agent
docker build -t topic10-entity-linking-agent .
docker run --rm -p 8000:8000 topic10-entity-linking-agent
```

容器内服务监听 `0.0.0.0:8000`，宿主机通过 `-p 8000:8000` 对外开放。

## 防火墙提醒

如果其他机器仍访问不到，请检查：

- Windows 防火墙是否允许 Python 或 8000 端口入站。
- 云服务器安全组是否放行 TCP 8000。
- 客户端访问时是否使用了真实服务器 IP，而不是 `0.0.0.0`。

## 环境变量

- `EL_HOST`: 默认 `0.0.0.0`。
- `EL_PORT`: 默认 `8000`。
- `EL_APP_NAME`
- `EL_APP_VERSION`
- `EL_DEFAULT_KB_ID`
- `EL_DEFAULT_KB_PATH`
- `EL_TRACES_DIR`
- `EL_TRACE_PREFIX`
- `EL_DEFAULT_TOP_K`

## 说明

当前框架不依赖外部在线服务，适合私有网络内运行。若后续接入大模型或向量库，建议把凭据改为环境变量注入，并保持可切换到本地替代实现。
