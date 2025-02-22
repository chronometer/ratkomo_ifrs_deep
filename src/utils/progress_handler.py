from typing import Any, Dict, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.panel import Panel
from rich.text import Text
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

class RichProgressHandler(BaseCallbackHandler):
    """A callback handler that shows rich progress bars and status updates."""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=self.console,
            transient=True,
        )
        self.current_task_id: Optional[TaskID] = None
        self.analysis_task_id: Optional[TaskID] = None
        self.synthesis_task_id: Optional[TaskID] = None
        self.total_steps = 100  # Approximate total steps
        self.current_step = 0
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a chain starts running."""
        chain_type = serialized.get("name", "Unknown Chain")
        if "Analysis" in chain_type:
            self.analysis_task_id = self.progress.add_task(
                f"[cyan]Running IFRS Analysis...", total=self.total_steps
            )
        elif "Synthesis" in chain_type:
            self.synthesis_task_id = self.progress.add_task(
                f"[green]Generating Report...", total=self.total_steps
            )
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a chain ends running."""
        if self.analysis_task_id:
            self.progress.update(self.analysis_task_id, completed=self.total_steps)
        if self.synthesis_task_id:
            self.progress.update(self.synthesis_task_id, completed=self.total_steps)
    
    def on_agent_action(self, action: AgentAction, **kwargs):
        """Called when an agent takes an action."""
        self.current_step += 1
        tool = action.tool
        tool_input = action.tool_input
        
        # Update progress based on the current phase
        if self.analysis_task_id and not self.synthesis_task_id:
            self.progress.update(self.analysis_task_id, completed=min(self.current_step, self.total_steps))
            self.console.print(f"[cyan]Using tool: {tool}")
        elif self.synthesis_task_id:
            self.progress.update(self.synthesis_task_id, completed=min(self.current_step, self.total_steps))
            self.console.print(f"[green]Using tool: {tool}")
    
    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Called when an agent finishes running."""
        if self.analysis_task_id:
            self.progress.update(self.analysis_task_id, completed=self.total_steps)
        if self.synthesis_task_id:
            self.progress.update(self.synthesis_task_id, completed=self.total_steps)
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        """Called when an LLM starts running."""
        self.current_task_id = self.progress.add_task(
            "[yellow]Thinking...",
            total=None,  # Indeterminate progress
        )
    
    def on_llm_end(self, response: LLMResult, **kwargs):
        """Called when an LLM ends running."""
        if self.current_task_id is not None:
            self.progress.remove_task(self.current_task_id)
            self.current_task_id = None
    
    def __enter__(self):
        """Start the progress tracking."""
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the progress tracking."""
        self.progress.stop()
