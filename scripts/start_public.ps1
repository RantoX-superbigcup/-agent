param(
  [string]$HostName = "0.0.0.0",
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$PythonExe = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
  $PythonExe = "python"
}

$env:EL_HOST = $HostName
$env:EL_PORT = [string]$Port
& $PythonExe .\main.py
