"""
Run context model for IFRS analysis
"""
from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class RunContext(BaseModel):
    """Context for a single analysis run"""
    task_id: str
    task_type: str
    start_time: str
    metadata: Optional[dict] = None
