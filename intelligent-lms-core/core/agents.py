import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from core.state import AgentState
from services.telemetry_ml import CognitiveState

# Initialize the LLM
# Note: Requires GOOGLE_API_KEY in environment
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def socratic_agent(state: AgentState) -> dict:
    """The Agent adapts its behavior based on the selected mode and cognitive state."""
    prereqs = state.get("prerequisites", [])
    prereqs_str = ", ".join(prereqs) if prereqs else "general programming concepts"
    
    cognitive_state = state.get("current_load_state", CognitiveState.FOCUSED)
    mode = state.get("mode", "Auto")
    
    # Determine if we should use Normal (direct answer) mode
    is_normal_mode = False
    if mode == "Normal":
        is_normal_mode = True
    elif mode == "Auto" and cognitive_state == CognitiveState.FRUSTRATED:
        is_normal_mode = True
        
    if is_normal_mode:
        system_prompt = (
            "You are a helpful programming tutor. "
            "Provide a direct, clear, and comprehensive answer to the student's question. "
            "Do not answer with another question. Explain the concepts clearly."
        )
    else:
        system_prompt = (
            "You are a Socratic tutor. Your goal is to guide the student to the answer by asking questions. "
            "DO NOT give the direct answer. "
            f"Focus your questions on these prerequisite concepts: {prereqs_str}. "
        )
        
        if cognitive_state == CognitiveState.FRUSTRATED:
            system_prompt += (
                "\n[High-Scaffolding Mode]: The student appears frustrated. Provide easier hints, "
                "be very encouraging, and break the problem down into smaller, simpler steps."
            )
        elif cognitive_state == CognitiveState.IDLE:
            system_prompt += (
                "\n[Engagement Mode]: The student appears idle. Try to engage them with a thought-provoking "
                "but accessible question to bring them back on track."
            )

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm.invoke(messages)
    return {"messages": [response]}

def memory_agent(state: AgentState) -> dict:
    """Compares user summary to graph nodes to identify missing nodes."""
    # Only run if we actually have some conversation history
    if not state["messages"]:
        return {"missing_nodes": []}
    
    last_message = state["messages"][-1].content.lower()
    
    # Simple check if the user is summarizing
    if "summary" in last_message or "so basically" in last_message or "to summarize" in last_message:
        graph_nodes = state.get("graph_nodes", {})
        mentioned_nodes = []
        for key, value in graph_nodes.items():
            name = value["name"].lower()
            if key in last_message or name in last_message:
                mentioned_nodes.append(key)
        
        missing = [key for key in graph_nodes.keys() if key not in mentioned_nodes]
        return {"missing_nodes": missing}
        
    return {"missing_nodes": []}
