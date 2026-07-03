from .base import BaseAgent
from typing import Dict, List

class ModeratorAgent(BaseAgent):
    def invoke(self, input_data: Dict) -> Dict:
        topic = input_data["topic"]
        messages = input_data.get("messages", [])
        personas = input_data.get("personas", [])

        context = "\n\n".join([f"{p['name']}: {p['profile'][:400]}..." for p in personas])

        prompt = f"""You are an expert moderator running a high-quality creative marketing roundtable.

        Topic: {topic}
        
        Experts Present:
        {context}
        
        Current Discussion:
        {[f"{m['role']}: {m['content'][:200]}" for m in messages[-6:]] if messages else 'No messages yet'}
        
        Your job:
        - Summarize key points
        - Keep the discussion focused and productive
        - Decide who should speak next
        - Push toward creative ideas (reels, ads, LinkedIn posts, campaigns)
        
        Respond as Moderator."""

        response = self.llm.invoke(prompt)
        
        return {
            "role": "Moderator",
            "content": response.content
        }