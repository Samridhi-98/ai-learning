# 1_simple_graph.py
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


# ── State schema ──────────────────────────────────────────────────────────────
# This is the shared context object — like your conversation_history from Phase 1
# BUT: add_messages appends automatically instead of you doing list.append()
# AND: it persists across nodes without you passing it manually
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Add checkpointer at compile time
memory = MemorySaver()

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(model="claude-haiku-4-5-20251001")


# ── Node ──────────────────────────────────────────────────────────────────────
# A node is just a function: takes State, returns partial State update
# LangGraph merges the return dict into full state automatically
def chatbot_node(state: State) -> State:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # add_messages appends this to history


# ── Graph construction ────────────────────────────────────────────────────────
# Think of this as wiring your Spring beans together
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_edge(START, "chatbot")  # entry point
graph_builder.add_edge("chatbot", END)  # exit point
graph = graph_builder.compile(checkpointer=memory)  # like Spring context refresh


# ── Run ───────────────────────────────────────────────────────────────────────
def main():
    print("LangGraph Chatbot — type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        config = {"configurable": {"thread_id": "user_samriddhi"}}
        result = graph.invoke(
            {"messages": [HumanMessage(user_input)]},
            config=config  # ← LangGraph uses this to load/save state
        )

        # last message in state = assistant response
        print(f"Assistant: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()