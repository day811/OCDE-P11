from abc import ABC, abstractmethod
from config import Config

class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, provider, temperature: float = 0.7):
        self.chat_model = Config.get_chat_model(provider)
        self.embed_model = Config.get_embed_model(provider)
        self.temperature = temperature

    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def embed(self, text) :
        """Generate embedding for text"""
        return text
