"""
OpenRouter model implementation for pydantic-ai
"""
from typing import List, Dict, Any, Optional
import logging
from pydantic_ai.models.openai import OpenAIModel

class OpenRouterModel(OpenAIModel):
    """OpenRouter model implementation with proper headers and configuration"""
    
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None
    ):
        super().__init__(model_name, base_url=base_url, api_key=api_key)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
        self._model_name = model_name
    
    async def request(
        self,
        messages: List[Dict[str, str]],
        settings: Optional[Dict[str, Any]] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Make a request to OpenRouter API"""
        if not self._client:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                default_headers={
                    'HTTP-Referer': 'http://localhost:8000',
                    'X-Title': 'IFRS Document Analyzer'
                }
            )
        
        logging.info(f"Sending request to OpenRouter with messages: {messages}")
        
        try:
            completion = await self._client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                temperature=parameters.get('temperature', 0.3) if parameters else 0.3,
                max_tokens=parameters.get('max_tokens', 2000) if parameters else 2000,
                top_p=parameters.get('top_p', 0.95) if parameters else 0.95
            )
            logging.info(f"OpenRouter response: {completion}")
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in OpenRouter request: {e}")
            raise e
