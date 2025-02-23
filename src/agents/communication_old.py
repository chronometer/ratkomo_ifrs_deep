"""
Communication structures for Chain of Agents implementation
Based on the Chain of Agents paper methodology
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator
from enum import Enum

from src.utils.validation import validate_timestamp

class CommunicationType(str, Enum):
    """Types of communication between agents"""
    EVIDENCE = "evidence"        # Factual information found
    ANALYSIS = "analysis"        # Analysis of information
    QUERY = "query"             # Request for clarification
    SYNTHESIS = "synthesis"     # Combined findings
    ERROR = "error"             # Error or issue report

class Evidence(BaseModel):
    """Evidence found in document segment"""
    source_location: str = Field(
        ...,
        description="Location in document where evidence was found",
        examples=["page_1.paragraph_2", "section_3.table_1"]
    )
    content: str = Field(
        ...,
        description="The actual evidence text",
        min_length=1,
        max_length=10000
    )
    confidence: float = Field(
        ...,
        description="Confidence score for the evidence (0-1)",
        ge=0.0,
        le=1.0,
        examples=[0.95, 0.87]
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Contextual information about the evidence"
    )

    class Config:
        json_schema_extra={
            "examples": [{
                "source_location": "page_1.paragraph_2",
                "content": "Company reported cash equivalents of $1M",
                "confidence": 0.95,
                "context": {"section": "current_assets"}
            }]
        }

class Analysis(BaseModel):
    """Analysis of financial information"""
    metrics: Dict[str, Any] = Field(
        ...,
        description="Financial metrics analyzed",
        examples=[{
            "revenue": 1000000,
            "profit_margin": 0.15,
            "current_ratio": 1.5
        }]
    )
    findings: List[str] = Field(
        ...,
        description="Analysis findings",
        min_length=1,
        examples=[
            "Revenue increased by 15% compared to previous year",
            "Current ratio below industry average"
        ]
    )
    implications: List[str] = Field(
        ...,
        description="Implications of findings",
        min_length=1,
        examples=[
            "Strong growth trajectory indicates market expansion",
            "May need to improve working capital management"
        ]
    )

    class Config:
        json_schema_extra={
            "examples": [{
                "metrics": {"revenue": 1000000, "profit_margin": 0.15},
                "findings": ["Revenue increased by 15%"],
                "implications": ["Strong growth trajectory"]
            }]
        }

class CommunicationUnit(BaseModel):
    """Core communication unit between agents"""
    id: str = Field(
        ...,
        description="Unique identifier for this communication",
        pattern=r'^[a-zA-Z0-9_-]+$'
    )
    type: CommunicationType = Field(
        ...,
        description="Type of communication"
    )
    source_agent: str = Field(
        ...,
        description="Agent that created this communication",
        min_length=1
    )
    target_agent: Optional[str] = Field(
        None,
        description="Intended recipient agent"
    )
    timestamp: str = Field(
        ...,
        description="When this communication was created",
        examples=["2025-02-21T15:00:00Z"],
        json_schema_extra={"format": "date-time"}
    )
    
    # Current segment information
    segment_id: str = Field(
        ...,
        description="ID of the current document segment",
        pattern=r'^seg_[0-9]+$'
    )
    segment_type: str = Field(
        ...,
        description="Type of segment (e.g., balance_sheet, income_statement)",
        examples=["balance_sheet", "income_statement", "cash_flow"]
    )
    segment_content: str = Field(
        ...,
        description="Content of current segment",
        min_length=1
    )
    
    # Evidence and analysis
    evidence: List[Evidence] = Field(
        default_factory=list,
        description="Evidence found in current segment"
    )
    analysis: Optional[Analysis] = Field(
        None,
        description="Analysis of current segment"
    )
    
    # Chain context
    previous_findings: List[Dict] = Field(
        default_factory=list,
        description="Relevant findings from previous agents"
    )
    accumulated_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Accumulated context through chain"
    )
    
    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )

    class Config:
        json_schema_extra={
            "examples": [{
                "id": "cu_001",
                "type": "evidence",
                "source_agent": "statement_analyzer",
                "timestamp": "2025-02-21T15:00:00Z",
                "segment_id": "seg_001",
                "segment_type": "balance_sheet",
                "segment_content": "...",
                "evidence": [{
                    "source_location": "page_1.paragraph_2",
                    "content": "Company reported cash equivalents of $1M",
                    "confidence": 0.95,
                    "context": {"section": "current_assets"}
                }]
            }]
        }

class ChainContext(BaseModel):
    """Context maintained throughout the chain"""
    document_id: str = Field(
        ...,
        description="ID of the document being processed",
        pattern=r'^doc_[a-zA-Z0-9_-]+$'
    )
    task_type: str = Field(
        ...,
        description="Type of analysis task",
        examples=["ifrs_compliance", "financial_analysis"]
    )
    start_time: str = Field(
        ...,
        description="When processing started",
        json_schema_extra={"format": "date-time"}
    )
    segments_processed: int = Field(
        default=0,
        description="Number of segments processed",
        ge=0
    )
    total_segments: int = Field(
        ...,
        description="Total number of segments",
        gt=0
    )
    global_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Global context available to all agents"
    )
    
    # IFRS specific context
    standards_in_scope: List[str] = Field(
        default_factory=list,
        description="IFRS standards being checked",
        min_length=1,
        examples=["IFRS 9", "IFRS 15", "IFRS 16"]
    )
    fiscal_period: Dict[str, str] = Field(
        ...,
        description="Fiscal period being analyzed",
        examples=[{
            "start": "2024-01-01",
            "end": "2024-12-31"
        }]
    )
    company_info: Dict[str, Any] = Field(
        ...,
        description="Company information",
        examples=[{
            "name": "Example Corp",
            "industry": "Technology",
            "jurisdiction": "US"
        }]
    )

    class Config:
        json_schema_extra={
            "examples": [{
                "document_id": "doc_001",
                "task_type": "ifrs_compliance",
                "start_time": "2025-02-21T15:00:00Z",
                "total_segments": 5,
                "standards_in_scope": ["IFRS 9", "IFRS 15"],
                "fiscal_period": {
                    "start": "2024-01-01",
                    "end": "2024-12-31"
                },
                "company_info": {
                    "name": "Example Corp",
                    "industry": "Technology"
                }
            }]
        }

class AgentResponse(BaseModel):
    """Response from an agent after processing"""
    success: bool = Field(
        ...,
        description="Whether processing was successful"
    )
    communication_unit: Optional[CommunicationUnit] = Field(
        None,
        description="Generated communication unit"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if processing failed",
        examples=["Invalid input format", "Processing timeout"]
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Processing metrics",
        examples=[{
            "processing_time_ms": 150,
            "tokens_processed": 1000,
            "confidence_score": 0.92
        }]
    )

    class Config:
        json_schema_extra={
            "examples": [{
                "success": True,
                "communication_unit": {
                    "id": "cu_001",
                    "type": "analysis",
                    "source_agent": "worker_1",
                    "timestamp": "2025-02-21T15:00:00Z",
                    "segment_id": "seg_001",
                    "segment_type": "balance_sheet",
                    "segment_content": "..."
                },
                "metrics": {
                    "processing_time_ms": 150,
                    "confidence_score": 0.92
                }
            }]
        }
