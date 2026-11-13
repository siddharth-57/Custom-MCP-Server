import asyncio
from pathlib import Path

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

model = ChatOllama(model="qwen3:8b", reasoning=False)  # reasoning=False = think off


async def main():
    # Keep the adapter open for the whole chat (explained below)
    async with MCPAdapter(Path("server.py")) as adapter:
        tools = await adapter.list_tools()
        print("Tools found:", [t.name for t in tools])

        agent = create_agent(model, tools, checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "chat-1"}}
        seen = 0

        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ("quit", "exit"):
                break
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]}, config
            )
            for m in result["messages"][seen:]:  # print only this turn's messages
                m.pretty_print()
            seen = len(result["messages"])


asyncio.run(main())