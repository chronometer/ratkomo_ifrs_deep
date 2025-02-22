"""
LLM Configuration using OpenRouter and LangWatch integration
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field
import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI

class LLMConfig(BaseModel):
    """Configuration for LLM settings"""
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    default_model: str = Field(default="anthropic/claude-3-opus", description="Default OpenRouter model")
    temperature: float = Field(default=0.1, description="Temperature for LLM responses")
    max_tokens: int = Field(default=4096, description="Maximum tokens for LLM responses")

class LLMManager:
    """Manager class for LLM interactions with tracing"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = ChatOpenAI(
            openai_api_key=config.openrouter_api_key,
            model=config.default_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
    
    async def get_completion(
        self, 
        prompt: str, 
        model: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Get completion from LLM with tracing
        
        Args:
            prompt: Input prompt
            model: Optional model override
            metadata: Optional metadata for tracing
        
        Returns:
            LLM response
        """
        if model:
            self.llm.model = model
            
        response = await self.llm.agenerate([prompt])
        return response.generations[0][0].text

def get_llm_config() -> LLMConfig:
    """Get LLM configuration from environment variables"""
    from dotenv import load_dotenv
    load_dotenv()
    
    return LLMConfig(
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        default_model=os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-3-opus"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )

# Example usage in agent:
"""
class DocumentProcessorAgent:
    def __init__(self):
        config = get_llm_config()
        self.llm_manager = LLMManager(config)
        
    async def process_document(self, content: str):
        metadata = {
            "agent": "document_processor",
            "document_type": "financial_statement"
        }
        
        response = await self.llm_manager.get_completion(
            prompt=f"Analyze this financial statement: {content}",
            metadata=metadata
        )
        return response
"""
