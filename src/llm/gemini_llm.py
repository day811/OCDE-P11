from typing import List
from src.llm.base import BaseLLM
from google import genai
import logging
from src.config import Config

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):

    NAME = "Gemnii AI"
    PROVIDER = "gemini"
    CHAT_MODEL = Config.get_chat_model(PROVIDER)
    EMBED_MODEL = Config.get_embed_model(PROVIDER)
    API_KEY = Config.get_api_key(PROVIDER)

    def __init__(self, temperature: float = 0.7):

        super().__init__(temperature)

        self.client = genai.Client(api_key=self.API_KEY)# type: ignore
        logger.info(f"GeminiLLM initialized - Chat: {self.CHAT_MODEL}, Embed: {self.EMBED_MODEL}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini API"""
        temperature  = temperature if temperature is not None else self.temperature
        response = self.client.models.generate_content(
            model= self.CHAT_MODEL,
            contents= prompt,
            config={
                "temperature": self.temperature,
                "max_output_tokens": 512,
            },
            )
        return response.text
    
    def embed(self, text: str):
        """Generate embeddings using Gemini embed API"""

        if isinstance(text, list):
            # Multiple texts → return list of embeddings
            response = self.client.models.embed_content( #type:ignore
                model=self.EMBED_MODEL,
                contents=text
            )
            return [embed.values for embed in response.embeddings]
        else:
            # Single text → return single embedding
            response = self.client.models.embed_content( #type:ignore
                model=self.EMBED_MODEL,
                contents=[text]
            )
            return response.embeddings[0].values

    @classmethod
    def get_langchain(cls,temperature: float = 0.7):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cls.CHAT_MODEL, # type: ignore
            google_api_key=cls.API_KEY,
            temperature=temperature
        )
