from typing import List, Union
from src.llm.base import BaseLLM
from mistralai import Mistral
import logging
from config import Config

logger = logging.getLogger(__name__)

class MistralLLM(BaseLLM):
    def __init__(self, temperature: float = 0.7):

        super().__init__(temperature)

        self.name = "Mistral AI"
        self.provider = "mistral"
        self.chat_model = Config.get_chat_model(self.provider)
        self.embed_model = Config.get_embed_model(self.provider)

        self.client = Mistral(api_key=Config.get_api_key(self.provider ))
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
    
    def embed(self, text) :
        """Generate embeddings using Mistral embed API"""
        if isinstance(text, list):
            # Multiple texts → return list of embeddings
            response = self.client.embeddings.create(
                model=self.embed_model,
                inputs=text
            )
            return [embed.embedding for embed in response.data]
        else:
            # Single text → return single embedding
            response = self.client.embeddings.create(
                model=self.embed_model,
                inputs=[text]
            )
            return response.data[0].embedding