param(
  [string]$HostName = "0.0.0.0",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m uvicorn entity_linking_agent.app:app --app-dir src --host $HostName --port $Port --reload
