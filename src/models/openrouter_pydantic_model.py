"""
OpenRouter model implementation for pydantic-ai
"""
from typing import Any, Dict, List, Optional
import os
import httpx
from pydantic_ai.models.base import BaseModel
from pydantic_ai.models.openai import OpenAIMessage

class OpenRouterModel(BaseModel):
    """OpenRouter model implementation"""
    
    def __init__(self, model: str):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://github.com/pydantic/pydantic-ai",
                "X-Title": "IFRS Analysis"
            }
        )
    
    async def request(
        self,
        messages: List[OpenAIMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> str:
        """Make a request to the OpenRouter API"""
        data = {
            "model": self.model,
            "messages": messages,
        }
        
        if temperature is not None:
            data["temperature"] = temperature
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        if top_p is not None:
            data["top_p"] = top_p
        if frequency_penalty is not None:
            data["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            data["presence_penalty"] = presence_penalty
        if stop is not None:
            data["stop"] = stop
        
        response = await self.client.post("/chat/completions", json=data)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
