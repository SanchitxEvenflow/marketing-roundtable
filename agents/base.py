from abc import ABC, abstractmethod
from typing import Dict


class BaseAgent(ABC):
    """All agents call the shared NVIDIA NIM gateway in core.llm — no per-agent
    LLM clients, so caps/retries/usage tracking stay in one place."""

    @abstractmethod
    def invoke(self, input_data: Dict) -> Dict:
        pass
