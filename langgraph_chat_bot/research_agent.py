from http.client import responses
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
    topic: str                    # input to all parallel nodes
    technical_analysis: str       # output of technical_node
    business_analysis: str        # output of business_node
    market_analysis: str            # output of market_node
    final_report: str             # output of combine_node

def input_node(state: State) -> State:
    # extract topic from last user message
    # store in state["topic"]
    return {"topic": state["messages"][-1].content}  # ← return only what changed

def technical_node(state: State) -> State:
    # ask LLM to analyze state["topic"] from technical angle
    # store result in state["technical_analysis"]
    return {"technical_analysis": llm.invoke([SystemMessage("You are a technical assistant"), HumanMessage(state["topic"])]).content}

def business_node(state: State) -> State:
    # same but business angle
    return {"business_analysis": llm.invoke([SystemMessage("You are a buisness assistant"), HumanMessage(state["topic"])]).content}

def market_node(state: State) -> State:
    # same but market trends angle
    return {"market_analysis": llm.invoke([SystemMessage("You are a market trends analyst"), HumanMessage(state["topic"])]).content}

def combine_node(state: State) -> State:
    final_report = (f"Technical Analysis: {state['technical_analysis']}\n\n"
                    f"Business Analysis: {state['business_analysis']}\n\n"
                    f"Market Analysis: {state['market_analysis']}")
    return {
        "final_report": final_report,
        "messages": [SystemMessage(final_report)]  # ← adds to messages
    }≠


graph_builder = StateGraph(State)

graph_builder.add_node("input_node", input_node)
graph_builder.add_node("technical_node", technical_node)
graph_builder.add_node("business_node", business_node)
graph_builder.add_node("market_node", market_node)
graph_builder.add_node("combine_node", combine_node)

# Fan-out: one node → three parallel nodes
graph_builder.add_edge(START, "input_node")
graph_builder.add_edge("input_node", "technical_node")
graph_builder.add_edge("input_node", "business_node")
graph_builder.add_edge("input_node", "market_node")

# Fan-in: three nodes → one combine node
graph_builder.add_edge("technical_node", "combine_node")
graph_builder.add_edge("business_node", "combine_node")
graph_builder.add_edge("market_node", "combine_node")

# Final node
graph_builder.add_edge("combine_node", END)


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