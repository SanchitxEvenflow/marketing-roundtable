from .base import BaseAgent
from typing import Dict

class CreativeDirectorAgent(BaseAgent):
    def invoke(self, input_data: Dict) -> Dict:
        topic = input_data["topic"]
        messages = input_data.get("messages", [])

        prompt = f"""You are the Creative Director synthesizing a marketing roundtable.

        Topic: {topic}
        
        Full Discussion:
        {[f"{m['role']}: {m['content'][:300]}" for m in messages]}
        
        Synthesize the best ideas into clear, actionable creative recommendations.
        Focus on:
        - Viral hooks
        - Reel concepts
        - LinkedIn post ideas
        - Ad scripts
        - Overall campaign direction

        Give a structured creative brief."""

        response = self.llm.invoke(prompt)
        
        return {
            "role": "Creative Director",
            "content": response.content
        }