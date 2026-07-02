from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, List
from neo4j_client import Neo4jClient

neo4j_client = Neo4jClient()
retriever = neo4j_client.retriever
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0.7)

class DebateState(TypedDict):
    topic: str
    round: int
    messages: List[dict]
    selected_personas: List[dict]

def moderator_node(state: DebateState):
    context = "\n\n".join([f"{p['name']}: {p['profile'][:500]}..." for p in state["selected_personas"]])
    
    prompt = f"""You are an expert moderator running a high-level marketing roundtable.

Topic: {state['topic']}

Relevant Experts: {context}

Current Discussion:
{state['messages'][-6:] if state['messages'] else 'No discussion yet'}

Summarize key points and decide who should speak next or if we can move toward consensus."""

    response = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [{"role": "Moderator", "content": response.content}]
    }

def persona_node(state: DebateState, persona_name: str):
    prompt = f"""You are {persona_name}. Stay completely in character based on your profile.

Topic: {state['topic']}
Previous discussion: {state['messages'][-4:] if state['messages'] else 'None'}

Give your strong opinion in your unique style."""

    response = llm.invoke(prompt)
    return {
        "messages": state["messages"] + [{"role": persona_name, "content": response.content}]
    }

# Build the Graph
workflow = StateGraph(DebateState)

workflow.add_node("moderator", moderator_node)

# Add dynamic persona nodes later

workflow.set_entry_point("moderator")
# More edges will be added dynamically

app = workflow.compile()