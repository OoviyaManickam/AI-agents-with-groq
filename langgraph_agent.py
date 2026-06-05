import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

load_dotenv()

# ── 1. DEFINE TOOLS ──────────────────────────────────────────────────────────
# @tool decorator tells LangChain this is a tool
# The docstring becomes the tool description sent to the LLM

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city."""
    data = {
        "berlin": "18°C, cloudy",
        "london": "12°C, rainy",
        "tokyo": "28°C, sunny"
    }
    return data.get(city.lower(), "Weather data not available")

@tool
def get_population(city: str) -> str:
    """Get the population of a given city."""
    data = {
        "berlin": "3.7 million",
        "london": "9 million",
        "tokyo": "14 million"
    }
    return data.get(city.lower(), "Population data not available")

tools = [get_weather, get_population]

# ── 2. DEFINE STATE ───────────────────────────────────────────────────────────
# State = what gets passed between nodes in the graph
# messages list grows as the agent runs — same concept as Phase 2

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # operator.add means new messages get appended

# ── 3. SET UP LLM ─────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.environ.get("GROQ_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)  # attaches tool descriptions to every LLM call

# ── 4. DEFINE NODES ───────────────────────────────────────────────────────────

def call_llm(state: AgentState):
    """Node: sends messages to LLM, gets response."""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def run_tools(state: AgentState):
    """Node: finds tool calls in last message, runs them, returns results."""
    last_message = state["messages"][-1]
    tool_map = {t.name: t for t in tools}
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_input = tool_call["args"]

        print(f"Tool called: {tool_name}")
        print(f"Inputs: {tool_input}")

        result = tool_map[tool_name].invoke(tool_input)

        print(f"Result: {result}\n")

        results.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )

    return {"messages": results}

# ── 5. DEFINE ROUTING LOGIC ───────────────────────────────────────────────────
# This function decides which node to go to next

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "run_tools"  # go to run_tools node
    return END  # finish

# ── 6. BUILD THE GRAPH ────────────────────────────────────────────────────────

graph = StateGraph(AgentState)

# Add nodes
graph.add_node("call_llm", call_llm)
graph.add_node("run_tools", run_tools)

# Set entry point
graph.set_entry_point("call_llm")

# Add edges
graph.add_conditional_edges(
    "call_llm",
    should_continue
)  # branches based on should_continue

graph.add_edge(
    "run_tools",
    "call_llm"
)  # after tools, always go back to LLM

# Compile
agent = graph.compile()

print(agent.get_graph().draw_ascii())

# ── 7. RUN ────────────────────────────────────────────────────────────────────

print("Starting LangGraph agent...\n")

result = agent.invoke({
    "messages": [
        HumanMessage(
            content="What's the weather and population in Tokyo?"
        )
    ]
})

print("Final answer:")
print(result["messages"][-1].content)