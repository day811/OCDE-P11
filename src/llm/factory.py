from src.llm.mistral_llm import MistralLLM
from src.llm.openai_llm import OpenAILLM
from src.llm.gemini_llm import GeminiLLM

from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatPerplexity 

import logging
from config import Config

logger = logging.getLogger(__name__)

class LLMFactory:
    """Factory for creating LLM instances"""
    
    PROVIDERS = {
        'mistral': MistralLLM,
        'openai': OpenAILLM,
        'gemini': GeminiLLM
    }
    
    @staticmethod
    def create_llm(temperature: float = 0.7, provider:str=""):
        """Create LLM instance based on provider"""
        if not provider:
            provider = Config.LLM_PROVIDER
        
        if provider not in LLMFactory.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(LLMFactory.PROVIDERS.keys())}")
        
        logger.info(f"Creating LLM instance - Provider: {provider}")
        return LLMFactory.PROVIDERS[provider](temperature=temperature )

def get_llm(temperature: float = 0.7, provider:str=""):
    """Convenience function to get LLM instance"""
    from config import Config
    

    temperature = temperature if temperature else Config.LLM_TEMPERATURE
    
    return LLMFactory.create_llm( temperature, provider)

def get_langchain_llm( temperature: float = 0.7):
    """
    Factory pour LangChain LLM multi-provider.
    
    Args:
        provider: 'mistral', 'openai', 'gemini', 'perplexity'
        api_key: Clé API correspondante
        model: Modèle spécifique (optionnel)
        temperature: 0.0-1.0
    """
    provider  = Config.LLM_PROVIDER
    model = Config.get_chat_model()
    api_key = Config.get_api_key()
    
    if provider == 'mistral':
        return ChatMistralAI(
            model=model, # type: ignore
            api_key=api_key,
            temperature=temperature
        )
    
    elif provider == 'openai':
        return ChatOpenAI(
            model=model,
            api_key=api_key, # type: ignore
            temperature=temperature
        )
    
    elif provider == 'gemini':
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature
        )
    else:
        raise ValueError(f"Provider non supporté: {provider}")
    