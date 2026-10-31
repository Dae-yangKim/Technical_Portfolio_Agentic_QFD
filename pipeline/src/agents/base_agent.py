from abc import ABC, abstractmethod
from openai import OpenAI
import settings

_client = OpenAI()

class BaseAgent(ABC):
    def __init__(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.model = settings.MODEL

    def call(self, messages: list[dict]) -> str:
        response = _client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_prompt}] + messages,
        )
        return response.choices[0].message.content

    @abstractmethod
    def run(self, state: dict) -> dict: ...