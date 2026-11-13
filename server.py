# @mcp.tool() exposes a function as a tool.
# The type hints become the tool’s input schema.
# The docstring becomes the description the LLM reads to decide when to use the tool. Vague docstrings lead to wrong tool choices.
# stdio transport: the client will launch this file as a subprocess and talk to it over stdin/stdout. 
# That means no ports or HTTP to set up, which is the simplest option for a local demo.

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("todo")

tasks: list[dict] = []  # in-memory, resets when the server restarts


@mcp.tool()
def add_task(title: str) -> str:
    """Add a new task to the to-do list."""
    task = {"id": len(tasks) + 1, "title": title, "done": False}
    tasks.append(task)
    return f"Added task {task['id']}: {title}"


@mcp.tool()
def list_tasks() -> list[dict]:
    """List all tasks with their id, title and done status."""
    return tasks


@mcp.tool()
def complete_task(task_id: int) -> str:
    """Mark a task as done using its id."""
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
            return f"Task {task_id} marked done"
    return f"No task with id {task_id}"


if __name__ == "__main__":
    mcp.run()  # stdio transport by default