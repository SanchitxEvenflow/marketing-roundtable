from .base import BaseAgent
from typing import Dict

class PersonaAgent(BaseAgent):
    def __init__(self, name: str, profile: str):
        super().__init__()
        self.name = name
        self.profile = profile[:8000]  # limit context

    def invoke(self, input_data: Dict) -> Dict:
        topic = input_data["topic"]
        previous_messages = input_data.get("messages", [])

        prompt = f"""You are {self.name}, a top Indian marketing expert and copywriter.
Stay 100% in character based on your unique style, tone, and philosophy.

Your Profile:
{self.profile}

Topic: {topic}

Previous Discussion:
{previous_messages[-4:] if previous_messages else 'This is the start.'}

Give your strong, opinionated response in your signature style."""

        response = self.llm.invoke(prompt)
        
        return {
            "role": self.name,
            "content": response.content
        }