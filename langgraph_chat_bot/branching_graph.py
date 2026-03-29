from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatAnthropic(model="claude-haiku-4-5-20251001")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    category: str

def classify_node(state: State) -> State:
    # Ask LLM to classify as technical/billing/general
    # Return {"category": "..."}
    print("→ Routed to: CLASSIFY")
    response = llm.invoke([
        SystemMessage(
            "Classify the user message as exactly one word: technical, billing, or general. Reply with one word only."),
        HumanMessage(state["messages"][-1].content)
    ])
    return {"category": response.content.strip().lower()}

def technical_node(state: State) -> State:
    print("→ Routed to: TECHNICAL")
    # Respond with technical system prompt
    response = llm.invoke([SystemMessage("You are a technical assistant"), HumanMessage(state["messages"][-1].content)])
    return {"messages": [response]}

def billing_node(state: State) -> State:
    print("→ Routed to: BILLING")
    # Respond with billing system prompt
    response = llm.invoke([SystemMessage("You are a billing assistant"), HumanMessage(state["messages"][-1].content)])
    return {"messages": [response]}

def general_node(state: State) -> State:
    print("→ Routed to: GENERAL")
    # Respond with general system prompt
    response = llm.invoke([SystemMessage("You are a general assistant"), HumanMessage(state["messages"][-1].content)])
    return {"messages": [response]}

def router(state: State) -> str:
    print("→ Routed to: ROUTER")
    return state["category"]    # ← just read what classify_node stored

graph_builder = StateGraph(State)

# Add all four nodes
graph_builder.add_node("classify", classify_node)
graph_builder.add_node("technical", technical_node)
graph_builder.add_node("billing", billing_node)
graph_builder.add_node("general", general_node)

# Wire edges
graph_builder.add_edge(START, "classify")
# Wire edges from router to END
graph_builder.add_edge("technical", END)
graph_builder.add_edge("billing", END)
graph_builder.add_edge("general", END)

# Conditional edge from classify → router decides next node
graph_builder.add_conditional_edges(
    "classify",
    router,
    {
        "technical": "technical",
        "billing": "billing",
        "general": "general"
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