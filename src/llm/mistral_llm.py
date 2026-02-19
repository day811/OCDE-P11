from typing import List, Union
from src.llm.base import BaseLLM
from mistralai import Mistral
import logging
from src.config import Config

logger = logging.getLogger(__name__)

class MistralLLM(BaseLLM):

    NAME = "Mistral AI"
    PROVIDER = "mistral"
    CHAT_MODEL = Config.get_chat_model(PROVIDER)
    EMBED_MODEL = Config.get_embed_model(PROVIDER)
    API_KEY = Config.get_api_key(PROVIDER)

    def __init__(self, temperature: float = 0.7):

        super().__init__(temperature)


        self.client = Mistral(api_key=self.API_KEY)
        logger.info(f"MistralLLM initialized - Chat: {self.CHAT_MODEL}, Embed: {self.EMBED_MODEL}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Mistral chat API"""
        temp = temperature if temperature is not None else self.temperature
        response = self.client.chat.complete(
            model=self.CHAT_MODEL,
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
                model=self.EMBED_MODEL,
                inputs=text
            )
            return [embed.embedding for embed in response.data]
        else:
            # Single text → return single embedding
            response = self.client.embeddings.create(
                model=self.EMBED_MODEL,
                inputs=[text]
            )
            return response.data[0].embedding
    
    @classmethod
    def get_langchain(cls,temperature: float = 0.7):
        from langchain_mistralai import ChatMistralAI

        return ChatMistralAI(
            model=cls.CHAT_MODEL, # type: ignore
            api_key=cls.API_KEY,
            temperature=temperature
        )
