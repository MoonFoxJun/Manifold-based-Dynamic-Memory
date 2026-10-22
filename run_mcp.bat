@echo off
rem Launch the MCP server with the Python on PATH.
rem Set SM_PYTHON to use a specific interpreter.
cd /d "%~dp0"
if defined SM_PYTHON (
  "%SM_PYTHON%" "%~dp0run_mcp.py" --data "%~dp0data\messages.jsonl" %*
) else (
  python "%~dp0run_mcp.py" --data "%~dp0data\messages.jsonl" %*
)
