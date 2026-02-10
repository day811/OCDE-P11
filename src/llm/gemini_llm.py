from typing import List
from src.llm.base import BaseLLM
from google import genai
import logging
from config import Config

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, temperature: float = 0.7):

        self.name = "Gemnii AI"
        self.provider = "mistral"
        super().__init__(self.provider, temperature)
    
        self.client = genai.Client(api_key=Config.get_api_key(self.provider ))# type: ignore
        logger.info(f"GeminiLLM initialized - Chat: {self.chat_model}, Embed: {self.embed_model}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini API"""
        temp = temperature if temperature is not None else self.temperature
        response = self.client.models.generate_content(
            model= self.chat_model,
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
                model=self.embed_model,
                contents=text
            )
            return [embed.values for embed in response.embeddings]
        else:
            # Single text → return single embedding
            response = self.client.models.embed_content( #type:ignore
                model=self.embed_model,
                contents=[text]
            )
            return response.embeddings[0].values
