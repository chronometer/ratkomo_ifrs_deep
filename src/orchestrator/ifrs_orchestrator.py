"""
IFRS Analysis Orchestrator using enhanced pydantic-ai integration
"""
from typing import List, Optional, Tuple
from pathlib import Path
import asyncio
import uuid
from datetime import datetime
import logging
import os
from textwrap import dedent

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from src.models.ifrs_models import (
    DocumentSegment,
    ComplianceAnalysis,
    FinalReport,
    IFRSStandard
)
from src.agents.ifrs_worker_agent import create_ifrs_worker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IFRSOrchestrator:
    """Orchestrates IFRS document analysis using enhanced Chain of Agents"""
    
    def __init__(
        self,
        num_workers: int = 3,
        model_name: str = "openai/gpt-4",
        min_segment_size: int = 500,  # Reduced to better handle financial sections
        max_segment_size: int = 4000,
        batch_size: int = 3,
        memory_type: str = "chat"
    ):
        """Initialize the IFRS orchestrator with enhanced configuration"""
        self.num_workers = num_workers
        self.min_segment_size = min_segment_size
        self.max_segment_size = max_segment_size
        self.batch_size = batch_size
        
        # Validate API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")
        
        # Initialize worker agents
        self.workers = [
            create_ifrs_worker(i, model_name, api_key=self.api_key)
            for i in range(num_workers)
        ]
        
        # Configure OpenRouter model for manager
        model = OpenAIModel(
            os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-3-opus"),
            base_url='https://openrouter.ai/api/v1',
            api_key=self.api_key,
        )
        
        # Initialize manager agent for synthesis
        self.manager = Agent(
            model=model,
            name="ifrs_manager",
            system_prompt="""As an IFRS compliance manager specializing in financial reporting:

            Your task is to synthesize segment analyses into a comprehensive IFRS compliance report.

            Focus areas:
            1. Financial Statement Presentation (IAS 1)
            2. Segment Reporting (IFRS 8)
            3. Revenue Recognition (IFRS 15)
            4. Financial Instruments (IFRS 9)
            5. Leases (IFRS 16)
            6. Disclosures and Transparency

            For each IFRS standard identified in the analyses:
            - Evaluate consistency of application across segments
            - Assess completeness of required disclosures
            - Identify potential compliance gaps
            - Suggest specific improvements

            Structure your response as:

            Overall Compliance: [compliant/non_compliant/partially_compliant]
            - Summary of compliance status with rationale

            Key Findings:
            - Major compliance strengths
            - Critical gaps or issues
            - Evidence from segment analyses

            Standards Analysis:
            - [Standard ID]: [Compliance assessment with evidence]

            Risk Areas:
            - Identified compliance risks
            - Missing disclosures
            - Areas needing attention

            Recommendations:
            - Specific actions to improve compliance
            - Priority improvements needed
            - Implementation suggestions

            Confidence: [0.0-1.0]
            - Based on completeness and quality of available information""",
            retries=3,
            model_settings={"temperature": float(os.getenv("LLM_TEMPERATURE", "0.2"))}
        )
    
    def _detect_segment_type_and_priority(self, title: str) -> Tuple[str, int]:
        """Detect the type of content and its priority based on its title"""
        title = title.lower()
        
        # Financial statement sections (highest priority)
        if any(x in title for x in ['liikevaihto', 'revenue', 'income', 'sales', 'financial', 'statement', 'tase', 'balance', 'sheet', 'profit', 'loss']):
            return 'financial_statements', 1
        
        # Notes and disclosures
        elif any(x in title for x in ['liitetiedot', 'notes', 'disclosure', 'accounting', 'policies']):
            return 'notes', 2
        
        # Segment information
        elif any(x in title for x in ['segmentti', 'segment', 'liiketoiminta', 'business']):
            return 'segments', 3
        
        # Risk and financial instruments
        elif any(x in title for x in ['riski', 'risk', 'rahoitus', 'financial', 'instruments', 'leases']):
            return 'risk_and_financial', 4
        
        # Governance and management
        elif any(x in title for x in ['hallinto', 'governance', 'johto', 'management', 'board']):
            return 'governance', 5
        
        # Other sections
        elif any(x in title for x in ['vastuullisuus', 'sustainability', 'esg', 'environmental']):
            return 'sustainability', 6
        elif any(x in title for x in ['strategia', 'strategy', 'vision', 'mission']):
            return 'strategic', 7
            
        return 'general', 8
    
    def segment_document(self, content: str) -> List[DocumentSegment]:
        """Enhanced document segmentation with unique IDs and improved content grouping"""
        segments = []
        current_segment = []
        current_size = 0
        current_type = None
        current_priority = 8  # Default to lowest priority
        
        # Split content into lines and group by sections
        lines = content.split('\n')
        section_starts = []
        
        # First pass: identify section boundaries
        for i, line in enumerate(lines):
            if line.isupper() and len(line.strip()) < 100:
                segment_type, priority = self._detect_segment_type_and_priority(line)
                section_starts.append((i, segment_type, priority))
        
        # Add end marker
        section_starts.append((len(lines), 'end', 8))
        
        # Second pass: create segments based on sections
        for i in range(len(section_starts) - 1):
            start_idx, segment_type, priority = section_starts[i]
            end_idx = section_starts[i + 1][0]
            
            # Get section content
            section_lines = lines[start_idx:end_idx]
            section_content = '\n'.join(section_lines)
            
            # Split large sections into smaller segments
            if len(section_content) > self.max_segment_size:
                # Split while preserving paragraph boundaries
                current_segment = []
                current_size = 0
                
                for line in section_lines:
                    current_segment.append(line)
                    current_size += len(line) + 1  # +1 for newline
                    
                    # Create new segment at paragraph boundary if size exceeds limit
                    if current_size >= self.max_segment_size and (not line.strip() or line.strip().endswith('.')):
                        segments.append(DocumentSegment(
                            content='\n'.join(current_segment),
                            segment_type=segment_type,
                            segment_id=str(uuid.uuid4())
                        ))
                        current_segment = []
                        current_size = 0
                
                # Add remaining lines as final segment
                if current_segment:
                    segments.append(DocumentSegment(
                        content='\n'.join(current_segment),
                        segment_type=segment_type,
                        segment_id=str(uuid.uuid4())
                    ))
            else:
                # Add entire section as one segment if small enough
                segments.append(DocumentSegment(
                    content=section_content,
                    segment_type=segment_type,
                    segment_id=str(uuid.uuid4())
                ))
        
        logger.info(f"Split document into {len(segments)} segments")
        return segments
    
    async def analyze_document(self, document_path: Path) -> FinalReport:
        """Analyze document using enhanced Chain of Agents approach"""
        try:
            # Read and segment document
            content = document_path.read_text(encoding='utf-8')
            segments = self.segment_document(content)
            
            # Process segments in batches
            batches = [segments[i:i + self.batch_size] 
                      for i in range(0, len(segments), self.batch_size)]
            
            all_analyses: List[ComplianceAnalysis] = []
            
            # Process each batch with progress tracking
            for i, batch in enumerate(batches, 1):
                logger.info(f"Processing batch {i}/{len(batches)}")
                
                # Create batch context
                batch_context = {
                    "batch_id": f"batch_{i}",
                    "batch_timestamp": datetime.utcnow().isoformat(),
                    "total_segments": len(segments),
                    "current_batch": i,
                    "total_batches": len(batches)
                }
                
                # Process batch segments in parallel
                analyses = await asyncio.gather(*[
                    self._analyze_segment(worker, segment, batch_context)
                    for worker, segment in zip(self.workers, batch)
                ])
                
                all_analyses.extend(analyses)
                logger.info(f"Completed batch {i} analysis")
            
            # Synthesize final report using manager agent
            logger.info("Synthesizing final report...")
            # Generate statistics
            stats = {
                "total_standards": sum(len(a.standards) for a in all_analyses),
                "compliant_segments": sum(1 for a in all_analyses if a.compliance_status == "compliant"),
                "non_compliant_segments": sum(1 for a in all_analyses if a.compliance_status == "non_compliant")
            }
            
            # Synthesize final report
            result = await self.manager.run(
                f"""Generate a final IFRS compliance report based on the following analyses:

                Document: {document_path}
                Total Segments: {len(segments)}
                
                Statistics:
                - Total Standards Referenced: {stats['total_standards']}
                - Compliant Segments: {stats['compliant_segments']}
                - Non-compliant Segments: {stats['non_compliant_segments']}
                
                Segment Analyses:
                {chr(10).join(f'Segment {i+1}: {a.dict()}' for i, a in enumerate(all_analyses))}
                
                Provide your report in a structured format that includes:
                - Overall compliance status (compliant/non_compliant/partially_compliant)
                - Key findings
                - Standards analysis with compliance levels
                - Risk areas
                - Recommendations
                - Confidence score (0.0 to 1.0)
                """
            )
            
            # Get response text
            response = result.data
            
            # Parse response into FinalReport model
            final_report = FinalReport(
                overall_compliance="partially_compliant",  # Default, will be updated
                key_findings=[],
                standards_analysis=[],
                recommendations=[],
                risk_areas=[],
                confidence=0.5,  # Default confidence
                timestamp=datetime.utcnow()
            )
            
            # Parse the response
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if line.lower().startswith('overall compliance:'):
                    try:
                        status = line.split(':')[1].strip().lower()
                        if status in ["compliant", "non_compliant", "partially_compliant"]:
                            final_report.overall_compliance = status
                    except:
                        pass
                elif line.lower().startswith('key findings:'):
                    current_section = 'findings'
                elif line.lower().startswith('standards analysis:'):
                    current_section = 'standards'
                elif line.lower().startswith('risk areas:'):
                    current_section = 'risks'
                elif line.lower().startswith('recommendations:'):
                    current_section = 'recommendations'
                elif line.lower().startswith('confidence:'):
                    try:
                        conf = float(line.split(':')[1].strip())
                        if 0.0 <= conf <= 1.0:
                            final_report.confidence = conf
                    except:
                        pass
                elif line.startswith('-') and current_section:
                    item = line[1:].strip()
                    if current_section == 'findings':
                        final_report.key_findings.append(item)
                    elif current_section == 'standards':
                        try:
                            if ':' in item:
                                std_id, details = item.split(':', 1)
                                final_report.standards_analysis.append(IFRSStandard(
                                    standard_id=std_id.strip(),
                                    title=details.strip(),
                                    compliance_level="unclear"
                                ))
                        except:
                            pass
                    elif current_section == 'risks':
                        final_report.risk_areas.append(item)
                    elif current_section == 'recommendations':
                        final_report.recommendations.append(item)
            
            return final_report
            
        except Exception as e:
            logger.error(f"Error analyzing document: {e}", exc_info=True)
            raise
    
    async def _analyze_segment(self, agent: Agent, segment: DocumentSegment, run_context: Optional[dict] = None) -> ComplianceAnalysis:
        """Analyze a document segment using an agent"""
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
            result = await agent.run(
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
            
            # Get response text
            response = result.data
            
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

    async def analyze_and_save_report(
        self,
        document_path: Path,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Analyze document and save report with enhanced formatting"""
        try:
            report = await self.analyze_document(document_path)
            
            # Setup output directory
            if output_dir is None:
                output_dir = Path('output')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"ifrs_analysis_{timestamp}.md"
            
            # Write report with enhanced formatting
            with output_path.open('w', encoding='utf-8') as f:
                f.write("# IFRS Compliance Analysis Report\n\n")
                f.write(f"**Generated:** {report.timestamp}\n")
                f.write(f"**Overall Confidence:** {report.confidence:.2%}\n\n")
                
                f.write("## Overall Compliance Status\n")
                f.write(f"**Status:** {report.overall_compliance}\n\n")
                
                f.write("## Key Findings\n")
                for finding in report.key_findings:
                    f.write(f"* {finding}\n")
                f.write("\n")
                
                f.write("## Risk Areas\n")
                for risk in report.risk_areas:
                    f.write(f"* {risk}\n")
                f.write("\n")
                
                f.write("## Standards Analysis\n")
                for standard in report.standards_analysis:
                    f.write(f"### {standard.standard_id}: {standard.title}\n")
                    f.write(f"**Compliance Level:** {standard.compliance_level}\n\n")
                    
                    if standard.findings:
                        f.write("**Findings:**\n")
                        for finding in standard.findings:
                            f.write(f"* {finding}\n")
                        f.write("\n")
                        
                    if standard.evidence:
                        f.write("**Evidence:**\n")
                        for evidence in standard.evidence:
                            f.write(f"* {evidence}\n")
                        f.write("\n")
                
                f.write("## Recommendations\n")
                for rec in report.recommendations:
                    f.write(f"* {rec}\n")
            
            logger.info(f"Report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving report: {e}", exc_info=True)
            raise
