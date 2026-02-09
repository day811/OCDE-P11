from typing import List
from src.llm.base import BaseLLM
from openai import OpenAI
import logging
from config import Config

logger = logging.getLogger(__name__)

class OpenAILLM(BaseLLM):
    def __init__(self, temperature: float = 0.7):

        super().__init__(Config.get_chat_model(), Config.get_embed_model(), temperature)
        self.name = "Open AI"
        self.client = OpenAI(api_key=Config.get_api_key())
        logger.info(f"OpenAILLM initialized - Chat: {self.chat_model}, Embed: {self.embed_model}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using OpenAI chat API"""
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        return response.choices.message.content # type: ignore
    
    def embed(self, text: str) :
        """Generate embeddings using OpenAI embed API"""
        response = self.client.embeddings.create(
            model=self.embed_model,
            input=text
        )
        if isinstance(text, list):
            # Multiple texts → return list of embeddings
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=text
            )
            return [embed.embedding for embed in response.data]
        else:
            # Single text → return single embedding
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=[text]
            )
            return response.data[0].embedding