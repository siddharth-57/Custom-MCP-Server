import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from ollama import AsyncClient

MODEL = "qwen3:8b"

# How to launch the MCP server: as a subprocess, talking over stdin/stdout
server_params = StdioServerParameters(command=sys.executable, args=["server.py"])


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()  # MCP handshake

# Part 1 (MCP): the client asks the server which tools it has, then converts their schemas into the format Ollama expects. 
# The LLM never talks to the MCP server directly. The client sits in the middle.
            # --- Part 1: MCP tool discovery ---
            mcp_tools = (await session.list_tools()).tools
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in mcp_tools
            ]
            print("Tools found:", [t.name for t in mcp_tools])


# Part 2 (agentic): this is the loop. The model either requests tool calls or gives a final answer. 
# Each tool result is appended to messages, so on the next turn the model sees what happened and decides the next step.

            # --- Part 2: the agent loop ---
            llm = AsyncClient()
            messages = []  # created once, outside the chat loop -> survives across turns (Multi-turn chat (agent memory))

# Why this works: the LLM itself is stateless and remembers nothing between calls. “Memory” is simply resending the full conversation history every time. Two loops now exist:
# Chat loop (outer): one iteration per user message.
# Agent loop (inner): one iteration per LLM call, which may involve several tool calls.


            while True:  # chat loop: one iteration per user message
                user_input = input("\nYou: ")
                if user_input.lower() in ("quit", "exit"):
                    break
                messages.append({"role": "user", "content": user_input})


                for _ in range(10):  # safety cap so a confused model can't loop forever

                # Qwen3 is a “hybrid reasoning” model. It can work in two modes:
                # Thinking on: before answering, the model writes out its reasoning step by step, like scratch work: “The user wants two tasks added. I should call add_task twice, then complete_task...”. Only after that does it give the answer or tool call.
                # Thinking off: it skips the scratch work and responds directly.
                    resp = await llm.chat(model=MODEL, messages=messages, tools=tools, think=True)
                    msg = resp.message
                    # if msg.thinking:
                    #     print("  (thinking)", msg.thinking)
                    messages.append(msg)

                    if not msg.tool_calls:  # no tool requested -> final answer
                        print("Agent:", msg.content)
                        break

                    for call in msg.tool_calls:
                        name, args = call.function.name, call.function.arguments
                        print(f"  -> calling {name}({args})")
                        result = await session.call_tool(name, args)  # executed via MCP
                        text = "\n".join(c.text for c in result.content if hasattr(c, "text"))
                        print(f"  <- {text}")
                        messages.append({"role": "tool", "content": text, "tool_name": name})


asyncio.run(main())