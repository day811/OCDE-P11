from src.llm.mistral_llm import MistralLLM
from src.llm.openai_llm import OpenAILLM
from src.llm.gemini_llm import GeminiLLM
import logging

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory for creating LLM instances"""
    
    PROVIDERS = {
        'mistral': MistralLLM,
        'openai': OpenAILLM,
        'gemini': GeminiLLM
    }
    
    @staticmethod
    def create(provider: str, api_key: str, chat_model: str = "", embed_model: str = "", temperature: float = 0.7):
        """Create LLM instance based on provider"""
        provider = provider.lower().strip()
        
        if provider not in LLMFactory.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(LLMFactory.PROVIDERS.keys())}")
        
        logger.info(f"Creating LLM instance - Provider: {provider}")
        return LLMFactory.PROVIDERS[provider](
            api_key=api_key,
            chat_model=chat_model,
            embed_model=embed_model,
            temperature=temperature
        )

def get_llm(provider: str = "", chat_model: str = "", embed_model: str = "", temperature: float = 0.7):
    """Convenience function to get LLM instance"""
    from config import Config
    
    provider = provider or Config.LLM_PROVIDER
    api_key = Config.LLM_API_KEY 
    if  not api_key:
        raise ValueError(f"Unknown API KEY. Set it up in .env")

    chat_model = chat_model or Config.LLM_CHAT_MODEL
    embed_model = embed_model or Config.LLM_EMBED_MODEL
    temperature = temperature if temperature is not None else Config.LLM_TEMPERATURE
    
    return LLMFactory.create(provider, api_key, chat_model, embed_model, temperature)
