from typing import List
from src.llm.base import BaseLLM
from google import genai
import logging

logger = logging.getLogger(__name__)

class GeminiLLM(BaseLLM):
    def __init__(self, api_key: str, chat_model: str = "", embed_model: str = "", temperature: float = 0.7):
        super().__init__(chat_model or "gemini-1.5-flash", embed_model or "models/embedding-001", temperature)
        genai.configure(api_key=api_key) # type: ignore
        self.chat_client = genai.GenerativeModel(self.chat_model)# type: ignore
        self.embed_model_name = self.embed_model
        logger.info(f"GeminiLLM initialized - Chat: {self.chat_model}, Embed: {self.embed_model}, Temp: {self.temperature}")
    
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text using Gemini API"""
        temp = temperature if temperature is not None else self.temperature
        response = self.chat_client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temp)
        )
        return response.text
    
    def embed(self, text: str) -> List[float]:
        """Generate embeddings using Gemini embed API"""
        response = genai.embed_content( # type: ignore
            model=self.embed_model_name,
            content=text
        )
        return response["embedding"]
