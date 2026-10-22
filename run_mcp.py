"""Entry point for MCP clients (Claude Desktop, Cursor, ...).

Clients launch this file with a Python interpreter; it puts the project root on
``sys.path`` and starts the stdio MCP server.

Example client config (Claude Desktop ``claude_desktop_config.json``)::

    {
      "mcpServers": {
        "semantic-memory": {
          "command": "D:\\\\path\\\\to\\\\.venv\\\\Scripts\\\\python.exe",
          "args": ["D:\\\\path\\\\to\\\\semantic_memory\\\\run_mcp.py",
                   "--data", "D:\\\\path\\\\to\\\\semantic_memory\\\\data\\\\messages.jsonl"]
        }
      }
    }
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic_memory.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    main()
