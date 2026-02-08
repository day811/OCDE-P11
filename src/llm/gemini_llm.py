from typing import List
from src.llm.base import BaseLLM
from google import genai
import logging
from config import Config

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, temperature: float = 0.7):

        super().__init__(Config.get_chat_model(), Config.get_embed_model(), temperature)
        genai.configure(api_key=Config.get_api_key()) # type: ignore
        self.client = genai.GenerativeModel(self.chat_model)# type: ignore
        logger.info(f"GeminiLLM initialized - Chat: {self.chat_model}, Embed: {self.embed_model}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini API"""
        temp = temperature if temperature is not None else self.temperature
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temp)
        )
        return response.text
    
    def embed(self, text: str):
        """Generate embeddings using Gemini embed API"""

        if isinstance(text, list):
            # Multiple texts → return list of embeddings
            response = genai.embed_content( #type:ignore
                model=self.embed_model,
                content=text
            )
            return [embed.embedding for embed in response.data]
        else:
            # Single text → return single embedding
            response = genai.embed_content( #type:ignore
                model=self.embed_model,
                content=[text]
            )
            return response.data[0].embedding
