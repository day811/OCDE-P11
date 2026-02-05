from typing import List
from src.llm.base import BaseLLM
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, chat_model: str = "", embed_model: str = "", temperature: float = 0.7):
        super().__init__(chat_model or "gpt-4o-mini", embed_model or "text-embedding-3-small", temperature)
        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAILLM initialized - Chat: {self.chat_model}, Embed: {self.embed_model}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = "") -> str:
        """Generate text using OpenAI chat API"""
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return response.choices.message.content
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings using OpenAI embed API"""
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=text
        )
        return response.data.embedding
