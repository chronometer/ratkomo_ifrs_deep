"""
Pydantic models for IFRS analysis using Pydantic model for LLM response validation
"""
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class DocumentSegment(BaseModel):
    """Represents a segment of the document with metadata"""
    content: str = Field(..., description="The text content of the segment")
    page_number: Optional[int] = Field(None, description="Page number where the segment appears")
    segment_type: Optional[str] = Field(None, description="Type of content in the segment")
    segment_id: str = Field(..., description="Unique identifier for the segment")

class IFRSStandard(BaseModel):
    """Represents an IFRS standard analysis"""
    standard_id: str = Field(..., description="IFRS standard identifier (e.g., 'IFRS 15')")
    title: str = Field(..., description="Title of the standard")
    findings: List[str] = Field(default_factory=list, description="Key findings related to this standard")
    compliance_level: Literal["compliant", "non_compliant", "partially_compliant", "unclear"] = Field(
        ..., description="Level of compliance with this standard"
    )
    evidence: List[str] = Field(default_factory=list, description="Evidence supporting the findings")

class ComplianceAnalysis(BaseModel):
    """Analysis result for a document segment"""
    segment_id: str = Field(..., description="ID of the analyzed segment")
    standards: List[IFRSStandard] = Field(default_factory=list)
    compliance_status: Literal["compliant", "non_compliant", "unclear"] = Field(
        ..., description="Overall compliance status for this segment"
    )
    issues: List[str] = Field(default_factory=list, description="Identified compliance issues")
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the analysis")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FinalReport(BaseModel):
    """Final synthesized report"""
    overall_compliance: Literal["compliant", "non_compliant", "partially_compliant"] = Field(
        ..., description="Overall compliance status"
    )
    key_findings: List[str] = Field(default_factory=list)
    standards_analysis: List[IFRSStandard] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    risk_areas: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
