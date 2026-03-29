# 3_react_agent.py
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ── Tools ─────────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input should be a valid Python math expression."""
    return eval(expression)

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Sunny, 28°C in {city}"

tools = [calculator, get_weather]

# ── LLM with tools bound ──────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")
llm_with_tools = llm.bind_tools(tools)   # ← tells LLM what tools exist

# ── State ─────────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ── Nodes ─────────────────────────────────────────────────────────────────────
def llm_node(state: State) -> State:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def router(state: State) -> str:
    # your implementation here
    last_message = state["messages"][-1]
    if last_message.tool_calls:  # ← type = tool
        return "tools"
    return "end"     # ← type = text

# ── Graph ─────────────────────────────────────────────────────────────────────
tool_node = ToolNode(tools)   # ← handles tool execution automatically

graph_builder = StateGraph(State)

graph_builder.add_node("llm_node", llm_node)
graph_builder.add_node("tool_node", tool_node)

graph_builder.add_edge(START, "llm_node")
graph_builder.add_edge("tool_node", "llm_node")
graph_builder.add_edge("llm_node", END)

graph_builder.add_conditional_edges(
    "llm_node",
    router,
    {
        "tools": "tool_node",   # ← map return value to node name
        "end": END
    }
)

def main():
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    config = {"configurable": {"thread_id": "test_session"}}

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        result = graph.invoke(
            {"messages": [HumanMessage(user_input)]},
            config=config
        )
        print(f"Assistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()

