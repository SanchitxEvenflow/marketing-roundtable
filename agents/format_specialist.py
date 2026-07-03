from .base import BaseAgent
from typing import Dict

class FormatSpecialistAgent(BaseAgent):
    def invoke(self, input_data: Dict, output_type: str = "linkedin") -> Dict:
        brief = input_data["brief"]
        
        prompt = f"""Convert the following creative brief into a **high-quality {output_type.upper()}** output.

        Brief:
        {brief}
        
        Requirements:
        - Make it ready to use
        - Use best practices for the format
        - Make it engaging and on-brand"""

        response = self.llm.invoke(prompt)
        
        return {
            "role": f"Format Specialist ({output_type})",
            "content": response.content
        }