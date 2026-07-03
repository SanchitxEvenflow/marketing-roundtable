from graph.debate_graph import app, retriever

topic = "Best LinkedIn content strategy for B2B SaaS companies in 2026"

# Get relevant personas
relevant = retriever.get_relevant_personas(topic, k=5)
print("Selected personas:", [p['name'] for p in relevant])

initial_state = {
    "topic": topic,
    "round": 1,
    "messages": [],
    "selected_personas": relevant
}

result = app.invoke(initial_state)

for msg in result["messages"]:
    print(f"\n{msg['role']}:")
    print(msg['content'][:400] + "..." if len(msg['content']) > 400 else msg['content'])