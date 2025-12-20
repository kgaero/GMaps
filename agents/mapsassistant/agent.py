"""Maps assistant agent using Google Maps MCP tooling."""

import os

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters


_GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
_MCP_STARTUP_TIMEOUT_SECONDS = 30.0

if not _GOOGLE_MAPS_API_KEY:
  raise ValueError(
    "GOOGLE_MAPS_API_KEY is required to start mapsassistant. "
    "Set it in .env as GOOGLE_MAPS_API_KEY=your_key."
  )

root_agent = LlmAgent(
  model="gemma-3-27b-it",
  name="maps_assistant_agent",
  instruction=(
    "Help the user with mapping, directions, and finding places using "
    "Google Maps tools."
  ),
  tools=[
    McpToolset(
      connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
          command="npx",
          args=[
            "-y",
            "@modelcontextprotocol/server-google-maps",
          ],
          # The MCP server expects the key as an environment variable.
          env={
            "GOOGLE_MAPS_API_KEY": _GOOGLE_MAPS_API_KEY,
          },
        ),
        timeout=_MCP_STARTUP_TIMEOUT_SECONDS,
      ),
    ),
  ],
)
