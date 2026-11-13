# Local MCP Agent

An AI agent that manages tasks and files by calling tools over the **Model Context Protocol (MCP)**. It runs fully locally on Ollama, with no API keys and no data leaving your machine.

The agent connects to two MCP servers at once: a custom task-management server and the official MCP filesystem server. It decides which tools to call, in what order, and chains results across servers to complete multi-step requests.

## Features

- **Custom MCP server** for task management (`add_task`, `list_tasks`, `complete_task`)
- **Multi-server orchestration**: combines the task server with the official filesystem server in a single request
- **Three agent implementations** of the same tool-calling loop:
  - `client.py`: plain Python agent loop using the MCP SDK and Ollama directly
  - `langgraph_client.py`: LangChain's prebuilt agent (`create_agent`)
  - `graph_client.py`: custom LangGraph `StateGraph` with explicit nodes and conditional edges
- **Conversation memory** across turns (manual history, and LangGraph checkpointing)
- **Scoped file access**: the agent can only write inside the `workspace/` folder

## Tech Stack

Python · MCP Python SDK (v2) · LangGraph · LangChain · Ollama (Qwen3 8B) · uv

## Project Structure

```
server.py            # Task-management MCP server
client.py            # Agent loop built from scratch
langgraph_client.py  # Agent using LangGraph's prebuilt agent
graph_client.py      # Agent as a custom LangGraph StateGraph (task + filesystem servers)
workspace/           # Folder the agent is allowed to write to
```

## Setup

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com), and Node.js (for the filesystem server).

```bash
git clone <repo-url>
cd <repo-folder>
uv sync
ollama pull qwen3:8b
mkdir workspace
```

## Usage

Make sure Ollama is running, then start any of the agents:

```bash
uv run graph_client.py       # task + filesystem servers
uv run langgraph_client.py   # task server, prebuilt agent
uv run client.py             # task server, plain Python loop
```

Type `quit` to exit.

## How It Works

1. The agent connects to each MCP server and discovers its tools.
2. The user's request and the tool schemas are sent to the LLM.
3. If the LLM requests tool calls, the agent runs them through MCP and sends the results back.
4. This repeats until the LLM responds without requesting a tool.
