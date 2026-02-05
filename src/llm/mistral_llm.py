from typing import List, Optional
from src.llm.base import BaseLLM
from mistralai import Mistral
import logging

logger = logging.getLogger(__name__)

class MistralLLM(BaseLLM):
    def __init__(self, api_key: str, chat_model: str = "", embed_model: str = "", temperature: float = 0.7):
        super().__init__(chat_model or "mistral-small", embed_model or "mistral-embed", temperature)
        self.client = Mistral(api_key=api_key)
        logger.info(f"MistralLLM initialized - Chat: {self.chat_model}, Embed: {self.embed_model}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Mistral chat API"""
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.complete(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        content = response.choices[0].message.content
        return str(content) if content else "" 
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings using Mistral embed API"""
        list_lext = [text] if isinstance(text, str) else text
        response = self.client.embeddings.create(
            model=self.embed_model,
            inputs=list_lext
        )
        return response.data[0].embedding or []
