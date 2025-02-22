"""
Optimized document analyzer for IFRS documents with reduced API calls
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from dataclasses import dataclass
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel, ModelRequestParameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentSegment(BaseModel):
    """Represents a segment of the document with metadata"""
    content: str = Field(..., description="The text content of the segment")
    page_number: Optional[int] = Field(None, description="Page number where the segment appears")
    segment_type: Optional[str] = Field(None, description="Type of content in the segment")

class AnalysisResult(BaseModel):
    """Structured analysis result with IFRS compliance details"""
    segments: List[DocumentSegment]
    key_findings: List[str]
    compliance_status: str
    relevant_standards: List[Dict[str, str]]
    recommendations: List[str]

class OpenRouterModel(OpenAIModel):
    def __init__(self, model_name, base_url=None, api_key=None):
        super().__init__(model_name, base_url=base_url, api_key=api_key)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
    
    async def request(self, messages, settings=None, parameters=None):
        # Add required OpenRouter headers
        if not self._client:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                default_headers={
                    'HTTP-Referer': 'http://localhost:8000',
                    'X-Title': 'Document Analyzer'
                }
            )
        
        # Log request details
        logging.info(f"Sending request with messages: {messages}")
        
        # Make the request
        try:
            # Convert message types to roles
            role_map = {
                'system': 'system',
                'human': 'user',
                'ai': 'assistant'
            }
            
            formatted_messages = []
            for m in messages:
                role = role_map.get(m.type, 'user')
                formatted_messages.append({
                    "role": role,
                    "content": m.content
                })
            
            completion = await self._client.chat.completions.create(
                model=self._model_name,
                messages=formatted_messages,
                temperature=0.3,
                max_tokens=2000,
                top_p=0.95
            )
            logging.info(f"OpenRouter response: {completion}")
            return completion
        except Exception as e:
            logging.error(f"Error making OpenRouter request: {e}")
            raise

class OptimizedAnalyzer:
    def __init__(self):
        """Initialize the optimized document analyzer"""
        load_dotenv()
        
        self.model = OpenRouterModel(
            model_name=os.getenv('DEFAULT_LLM_MODEL'),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        
        # Configuration for optimized processing
        self.min_segment_size = 1000  # Minimum characters per segment
        self.max_segment_size = 4000  # Maximum to stay within token limits
        self.batch_size = 3  # Number of segments to analyze in one batch
        
    def segment_document(self, content: str) -> List[DocumentSegment]:
        """Smart document segmentation that combines related content"""
        segments = []
        current_segment = []
        current_size = 0
        current_type = None
        
        # Split into initial chunks by section markers
        lines = content.split('\n')
        
        for line in lines:
            # Detect section type based on content
            if line.isupper() and len(line.strip()) < 100:
                segment_type = self._detect_segment_type(line)
                
                # If we have a full segment, save it
                if current_size >= self.min_segment_size and current_type != segment_type:
                    segments.append(DocumentSegment(
                        content='\n'.join(current_segment),
                        segment_type=current_type
                    ))
                    current_segment = []
                    current_size = 0
                
                current_type = segment_type
            
            current_segment.append(line)
            current_size += len(line)
            
            # Split if segment gets too large
            if current_size >= self.max_segment_size:
                segments.append(DocumentSegment(
                    content='\n'.join(current_segment),
                    segment_type=current_type
                ))
                current_segment = []
                current_size = 0
        
        # Add final segment
        if current_segment:
            segments.append(DocumentSegment(
                content='\n'.join(current_segment),
                segment_type=current_type
            ))
        
        return segments

    def _detect_segment_type(self, title: str) -> str:
        """Detect the type of content in a segment based on its title"""
        title = title.lower()
        if any(x in title for x in ['revenue', 'income', 'sales']):
            return 'revenue_recognition'
        elif any(x in title for x in ['lease', 'rental', 'property']):
            return 'leasing'
        elif any(x in title for x in ['financial', 'instrument', 'asset']):
            return 'financial_instruments'
        elif any(x in title for x in ['segment', 'reporting']):
            return 'segment_reporting'
        return 'general'

    async def analyze_batch(self, segments: List[DocumentSegment]) -> AnalysisResult:
        """Analyze a batch of segments in a single API call"""
        # Combine segments with clear separators
        combined_content = "\n=== SEGMENT SEPARATOR ===\n".join(
            [f"[Type: {s.segment_type}]\n{s.content}" for s in segments]
        )
        
        messages = [
            SystemMessage(content="""You are an IFRS expert analyzing financial documents. 
            Analyze the following document segments and provide a comprehensive analysis covering:
            1. Key compliance findings for each IFRS standard
            2. Specific standard references (e.g., IFRS 15.31)
            3. Clear recommendations for improvement
            4. Overall compliance status
            
            Focus particularly on:
            - Revenue recognition (IFRS 15)
            - Financial instruments (IFRS 9)
            - Leasing arrangements (IFRS 16)
            - Segment reporting (IFRS 8)"""),
            HumanMessage(content=f"Analyze these document segments:\n\n{combined_content}")
        ]
        
        # Make single API call for batch
        response = await self.model.request(messages)
        
        # Process response into structured format
        return self._process_analysis_response(response, segments)

    def _process_analysis_response(self, response, segments: List[DocumentSegment]) -> AnalysisResult:
        """Process the API response into a structured analysis result"""
        # Extract content from response
        if hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content
        else:
            logging.error(f"Unexpected response format: {response}")
            content = str(response)
        
        logging.info(f"Processing content: {content}")
        
        # Parse the content into structured format
        findings = self._extract_findings(content)
        status = self._extract_compliance_status(content)
        standards = self._extract_standards(content)
        recommendations = self._extract_recommendations(content)
        
        return AnalysisResult(
            segments=segments,
            key_findings=findings,
            compliance_status=status,
            relevant_standards=standards,
            recommendations=recommendations
        )

    def generate_report(self, results: List[AnalysisResult]) -> str:
        """Generate final analysis report in markdown format"""
        report = []
        report.append("# IFRS Compliance Analysis Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        report.append("## Executive Summary")
        report.append("Based on the analysis of the provided document segments, the following key points were identified:\n")

        # Key Findings
        report.append("## Key Findings")
        for result in results:
            for finding in result.key_findings:
                report.append(f"- {finding}")
        report.append("\n")

        # Compliance Status
        report.append("## Compliance Status")
        for result in results:
            report.append(f"Overall status: {result.compliance_status}\n")
            report.append("Detailed status by standard:\n")

        # Key Standards
        report.append("## Relevant IFRS Standards")
        for result in results:
            for standard in result.relevant_standards:
                report.append(f"### {standard.get('standard', '')}")
                if standard.get('section'):
                    report.append(f"Section: {standard['section']}")
                report.append(f"Description: {standard.get('description', '')}\n")

        # Recommendations
        report.append("## Recommendations")
        for result in results:
            for recommendation in result.recommendations:
                report.append(f"- {recommendation}")
        report.append("\n")

        # Detailed Analysis
        report.append("## Detailed Analysis")
        for i, result in enumerate(results, 1):
            report.append(f"### Document Segment {i}\n")
            report.append("#### Content")
            for segment in result.segments:
                report.append(f"```\n{segment.content}\n```\n")
            report.append("#### Analysis")
            report.append("##### Key Points")
            for finding in result.key_findings:
                report.append(f"- {finding}")
            report.append("\n##### Standards Referenced")
            for standard in result.relevant_standards:
                report.append(f"- {standard.get('standard', '')}: {standard.get('description', '')}")
            report.append("\n##### Recommendations")
            for recommendation in result.recommendations:
                report.append(f"- {recommendation}")
            report.append("\n")

        return "\n".join(report)

    async def analyze_document(self, document_path: Path) -> str:
        """Analyze a document and generate a report"""
        # Read document
        with open(document_path, 'r') as f:
            content = f.read()
        
        # Segment document
        segments = self.segment_document(content)
        logging.info(f"Document divided into {len(segments)} optimized segments")
        
        # Analyze segments in batches
        results = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            result = await self.analyze_batch(batch)
            results.append(result)
            logging.info(f"Processed batch {len(results)}/{(len(segments) + self.batch_size - 1) // self.batch_size}")
        
        # Generate report
        return self.generate_report(results)
        # Read document
        with open(document_path, 'r') as f:
            content = f.read()
        
        # Smart segmentation
        segments = self.segment_document(content)
        logging.info(f"Document divided into {len(segments)} optimized segments")
        
        # Process in batches
        results = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            result = await self.analyze_batch(batch)
            results.append(result)
            logging.info(f"Processed batch {i//self.batch_size + 1}/{(len(segments) + self.batch_size - 1)//self.batch_size}")
        
        # Generate final report
        return self._generate_report(results)

    def _extract_findings(self, content: str) -> List[str]:
        """Extract key findings from analysis response"""
        findings = []
        try:
            # Try to find a JSON block in the content
            import re
            json_match = re.search(r'\{[^}]*"key_findings":[^}]*\}', content)
            if json_match:
                import json
                data = json.loads(json_match.group(0))
                findings = data.get('key_findings', [])
            else:
                # Extract findings from markdown-style lists
                for line in content.split('\n'):
                    if line.strip().startswith('*') and 'Key Findings' not in line:
                        findings.append(line.strip('* ').strip())
        except Exception as e:
            logging.error(f"Error extracting findings: {e}")
        return findings

    def _extract_compliance_status(self, content: str) -> str:
        """Extract overall compliance status"""
        try:
            # Try to find compliance status in JSON
            import re
            json_match = re.search(r'\{[^}]*"compliance_status":[^}]*\}', content)
            if json_match:
                import json
                data = json.loads(json_match.group(0))
                return data.get('compliance_status', 'needs_review')
            else:
                # Look for status in text
                status_match = re.search(r'compliance status[:\s]+(\w+)', content, re.IGNORECASE)
                if status_match:
                    return status_match.group(1).lower()
        except Exception as e:
            logging.error(f"Error extracting compliance status: {e}")
        return 'needs_review'

    def _extract_standards(self, content: str) -> List[Dict[str, str]]:
        """Extract relevant IFRS standards"""
        standards = []
        try:
            # Try to find standards in JSON
            import re
            json_match = re.search(r'\{[^}]*"relevant_standards":[^}]*\}', content)
            if json_match:
                import json
                data = json.loads(json_match.group(0))
                standards = data.get('relevant_standards', [])
            else:
                # Extract standards from text
                standard_matches = re.finditer(r'IFRS\s+(\d+)[.:]?\s*([\d.]+)?\s*[:-]\s*([^\n]+)', content)
                for match in standard_matches:
                    standards.append({
                        'standard': f'IFRS {match.group(1)}',
                        'section': match.group(2) if match.group(2) else '',
                        'description': match.group(3).strip()
                    })
        except Exception as e:
            logging.error(f"Error extracting standards: {e}")
        return standards

    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from analysis"""
        recommendations = []
        try:
            # Try to find recommendations in JSON
            import re
            json_match = re.search(r'\{[^}]*"recommendations":[^}]*\}', content)
            if json_match:
                import json
                data = json.loads(json_match.group(0))
                recommendations = data.get('recommendations', [])
            else:
                # Extract recommendations from text
                in_recommendations = False
                for line in content.split('\n'):
                    if 'Recommendations' in line:
                        in_recommendations = True
                        continue
                    if in_recommendations and line.strip().startswith(('*', '-')):
                        recommendations.append(line.strip('* -').strip())
                    elif in_recommendations and line.strip() and not line.strip().startswith(('*', '-')):
                        in_recommendations = False
        except Exception as e:
            logging.error(f"Error extracting recommendations: {e}")
        return recommendations

    def _generate_report(self, results: List[AnalysisResult]) -> str:
        """Generate final analysis report in markdown format"""
        # Combine all results into a comprehensive report
        report_parts = [
            "# IFRS Compliance Analysis Report",
            f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Executive Summary",
            self._generate_executive_summary(results),
            "\n## Detailed Analysis",
            self._generate_detailed_analysis(results)
        ]
        
        return "\n".join(report_parts)

    def _generate_executive_summary(self, results: List[AnalysisResult]) -> str:
        """Generate executive summary from all results"""
        summary_parts = []
        for result in results:
            summary_parts.extend([
                "## Key Findings",
                "\n".join(f"- {finding}" for finding in result.key_findings),
                "",
                "## Compliance Status",
                f"Overall status: {result.compliance_status}",
                "",
                "## Key Standards",
                "\n".join(f"- {std['standard']}: {std.get('description', '')}" for std in result.relevant_standards),
                "",
                "## Recommendations",
                "\n".join(f"- {rec}" for rec in result.recommendations),
                ""
            ])
        return "\n".join(summary_parts)

    def _generate_detailed_analysis(self, results: List[AnalysisResult]) -> str:
        """Generate detailed analysis section"""
        analysis_parts = []
        for i, result in enumerate(results):
            analysis_parts.extend([
                f"### Segment {i+1}",
                "",
                "#### Content",
                "```",
                "\n".join(s.content for s in result.segments),
                "```",
                "",
                "#### Analysis",
                "**Key Findings:**",
                "\n".join(f"- {finding}" for finding in result.key_findings),
                "",
                "**Relevant IFRS Standards:**",
                "\n".join(f"- {std['standard']} {std.get('section', '')}: {std.get('description', '')}" for std in result.relevant_standards),
                "",
                "**Recommendations:**",
                "\n".join(f"- {rec}" for rec in result.recommendations),
                "",
                "---",
                ""
            ])
        return "\n".join(analysis_parts)
