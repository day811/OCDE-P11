from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, chat_model: str = "", embed_model: str = "", temperature: float = 0.7):
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.temperature = temperature
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def embed(self, text) :
        """Generate embedding for text"""
        return text
