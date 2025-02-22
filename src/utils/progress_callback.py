from typing import Any, Dict, List, Optional
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult
import sys

class ProgressCallback(BaseCallbackHandler):
    """A callback handler that shows progress using simple print statements."""

    
    def __init__(self):
        self.current_phase = None
        self.step_count = 0
        self.last_phase = None
        self.total_tasks = {
            "analysis": 25,  # Approximate tasks for analysis phase
            "synthesis": 10   # Approximate tasks for synthesis phase
        }
        print("\n🔄 Starting IFRS Analysis...")
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when an LLM starts running."""
        pass
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when an LLM ends running."""
        pass
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a chain starts running."""
        # Progress is now handled directly in the orchestrator
        pass
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a chain ends running."""
        # Progress is now handled directly in the orchestrator
        pass
    
    def on_agent_action(self, action: AgentAction, **kwargs):
        """Called when an agent takes an action."""
        # Progress is now tracked in on_chain_start
        pass
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Called when an agent finishes running."""
        # Progress is now handled directly in the orchestrator
        pass
