# Hand-built LangGraph StateGraph
# This builds the same agent without create_agent. You define the pieces yourself:

# State: what’s passed between steps (here, the message list)
# Nodes: the steps (call the LLM, run tools)
# Edges: what runs next, including one conditional edge

import asyncio
from pathlib import Path

from langchain.mcp import MCPAdapter
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph


async def main():
    async with MCPAdapter(Path("server.py")) as adapter:
        tools = await adapter.list_tools()
        tools_by_name = {t.name: t for t in tools}

        # bind_tools = attach the tool schemas to every LLM call
        model = ChatOllama(model="qwen3:8b", reasoning=False).bind_tools(tools)

        # Node 1: ask the LLM what to do next
        async def call_llm(state: MessagesState):
            response = await model.ainvoke(state["messages"])
            return {"messages": [response]}

        # Node 2: run every tool the LLM requested
        async def run_tools(state: MessagesState):
            last = state["messages"][-1]
            results = []
            for call in last.tool_calls:
                tool = tools_by_name[call["name"]]
                results.append(await tool.ainvoke(call))  # returns a ToolMessage
            return {"messages": results}

        # Conditional edge: tool calls requested -> "tools", otherwise finish
        def should_continue(state: MessagesState):
            return "tools" if state["messages"][-1].tool_calls else END

        # Wire up the graph
        graph = StateGraph(MessagesState)
        graph.add_node("llm", call_llm)
        graph.add_node("tools", run_tools)
        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", should_continue, ["tools", END])
        graph.add_edge("tools", "llm")  # after tools, always go back to the LLM
        agent = graph.compile(checkpointer=InMemorySaver())

        print(agent.get_graph().draw_mermaid())  # prints the graph's structure

        # Same chat loop as before
        config = {"configurable": {"thread_id": "chat-1"}}
        seen = 0
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ("quit", "exit"):
                break
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": user_input}]}, config
            )
            for m in result["messages"][seen:]:
                m.pretty_print()
            seen = len(result["messages"])


asyncio.run(main())