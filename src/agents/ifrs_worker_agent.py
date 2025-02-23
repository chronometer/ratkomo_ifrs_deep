"""
IFRS Worker Agent Implementation using pydantic-ai
"""
from typing import List, Optional
import logging
import os
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from src.models.ifrs_models import (
    DocumentSegment,
    ComplianceAnalysis,
    IFRSStandard
)

logger = logging.getLogger(__name__)

def create_ifrs_worker(worker_id: int, model_name: Optional[str] = None, api_key: Optional[str] = None) -> Agent:
    """Create a worker agent specialized in IFRS analysis"""
    # Configure OpenRouter model
    model = OpenAIModel(
        model_name or os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-3-opus"),
        base_url='https://openrouter.ai/api/v1',
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
    )
    
    return Agent(
        model=model,
        name=f"ifrs_worker_{worker_id}",
        system_prompt="""As an IFRS expert specializing in financial statement analysis:

        Analyze the provided document segment for IFRS compliance, focusing on:
        1. Financial statements and metrics (revenue, EBITA, assets, liabilities)
        2. Accounting policies and disclosures
        3. Segment reporting (IAS 14/IFRS 8)
        4. Revenue recognition (IFRS 15)
        5. Financial instruments (IFRS 9)
        6. Leases (IFRS 16)
        
        For each identified IFRS standard:
        - Cite specific evidence from the text
        - Assess compliance level
        - Note any missing required disclosures
        - Suggest improvements if needed
        
        Structure your response as:
        Standards:
        - [Standard ID]: [Description]
        
        Compliance Status: [compliant/non_compliant/unclear]
        
        Findings:
        - [Key findings with evidence]
        
        Issues:
        - [Compliance issues or missing disclosures]
        
        Recommendations:
        - [Specific improvements needed]
        
        Confidence: [0.0-1.0]
        """,
        retries=3,  # Auto-retry on rate limits
        model_settings={"temperature": float(os.getenv("LLM_TEMPERATURE", "0.2"))}
    )

async def analyze_segment(agent: Agent, segment: DocumentSegment, run_context: Optional[dict] = None) -> ComplianceAnalysis:
    """Analyze a document segment for IFRS compliance"""
    try:
        # Prepare the context for analysis
        context = {
            "segment_id": segment.segment_id,
            "segment_type": segment.segment_type,
            "content": segment.content
        }
        if run_context:
            context.update(run_context)
            
        # Run analysis with response validation
        response = await agent.run(
            f"""Analyze this document segment for IFRS compliance:
            
            Content:
            {segment.content}
            
            Context:
            - Segment ID: {context['segment_id']}
            - Type: {context['segment_type']}
            
            Provide your analysis in a structured format that includes:
            - Standards identified and their compliance status
            - Key findings and evidence
            - Issues and recommendations
            - Confidence score (0.0 to 1.0)
            """
        )
        
        # Parse response into ComplianceAnalysis model
        analysis = ComplianceAnalysis(
            segment_id=segment.segment_id,
            standards=[],  # Will be populated from response
            compliance_status="unclear",  # Default, will be updated
            issues=[],
            key_findings=[],
            recommendations=[],
            confidence=0.5,  # Default confidence
            timestamp=datetime.utcnow()
        )
        
        # Parse the response and update the analysis
        lines = response.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.lower().startswith('standards:'):
                current_section = 'standards'
            elif line.lower().startswith('compliance status:'):
                try:
                    status = line.split(':')[1].strip().lower()
                    if status in ["compliant", "non_compliant", "unclear"]:
                        analysis.compliance_status = status
                except:
                    pass
            elif line.lower().startswith('issues:'):
                current_section = 'issues'
            elif line.lower().startswith('findings:'):
                current_section = 'findings'
            elif line.lower().startswith('recommendations:'):
                current_section = 'recommendations'
            elif line.lower().startswith('confidence:'):
                try:
                    conf = float(line.split(':')[1].strip())
                    if 0.0 <= conf <= 1.0:
                        analysis.confidence = conf
                except:
                    pass
            elif line.startswith('-') and current_section:
                item = line[1:].strip()
                if current_section == 'standards':
                    try:
                        if ':' in item:
                            std_id, details = item.split(':', 1)
                            analysis.standards.append(IFRSStandard(
                                standard_id=std_id.strip(),
                                title=details.strip(),
                                compliance_level="unclear"
                            ))
                    except:
                        pass
                elif current_section == 'issues':
                    analysis.issues.append(item)
                elif current_section == 'findings':
                    analysis.key_findings.append(item)
                elif current_section == 'recommendations':
                    analysis.recommendations.append(item)
        
        logger.info(f"Successfully analyzed segment {segment.segment_id}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing segment {segment.segment_id}: {e}", exc_info=True)
        # Return minimal valid analysis on error
        return ComplianceAnalysis(
            segment_id=segment.segment_id,
            standards=[],
            compliance_status="unclear",
            issues=["Analysis failed due to technical error"],
            key_findings=[],
            recommendations=["Review segment manually due to analysis failure"],
            confidence=0.0,
            timestamp=datetime.utcnow()
        )
