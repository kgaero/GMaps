"""Tests for the maps assistant agent."""

import importlib

from google.adk.tools.mcp_tool import McpToolset


def _load_agent(monkeypatch, api_key):
  """Reloads the agent module after setting the API key environment."""
  if api_key is None:
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
  else:
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", api_key)
  module = importlib.import_module("agents.mapsassistant.agent")
  return importlib.reload(module)


def test_root_agent_with_api_key(monkeypatch):
  """Ensures the agent is configured when a key is present."""
  module = _load_agent(monkeypatch, "test-key")
  root_agent = module.root_agent

  assert root_agent.name == "maps_assistant_agent"
  assert root_agent.model == "gemma-3-27b-it"
  assert "Google Maps tools" in root_agent.instruction
  assert len(root_agent.tools) == 1

  toolset = root_agent.tools[0]
  assert isinstance(toolset, McpToolset)
  assert toolset._connection_params.server_params.command == "npx"
  assert (
    toolset._connection_params.server_params.env["GOOGLE_MAPS_API_KEY"]
    == "test-key"
  )
  assert toolset._connection_params.timeout == 30.0


def test_root_agent_without_api_key(monkeypatch):
  """Ensures the agent errors when the key is missing."""
  try:
    _load_agent(monkeypatch, None)
  except ValueError as exc:
    assert "GOOGLE_MAPS_API_KEY is required" in str(exc)
  else:
    raise AssertionError("Expected ValueError when key is missing.")
