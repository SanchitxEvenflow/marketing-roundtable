from abc import ABC, abstractmethod
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, model="gemini-1.5-pro"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)

    @abstractmethod
    def invoke(self, input_data: Dict) -> Dict:
        pass