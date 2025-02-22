"""
Basic document analyzer for IFRS documents
"""
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import pypdf
import re
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from typing import Literal

class AnalysisContext(BaseModel):
    """Context for document analysis with validation"""
    document_path: Path = Field(..., description="Path to the document to analyze")
    output_dir: Path = Field(..., description="Directory to store analysis outputs")
    
    class Config:
        frozen = True

class IFRSStandard(BaseModel):
    """Represents an IFRS standard reference"""
    standard: str = Field(..., description="IFRS standard number (e.g., IFRS 15)")
    section: str = Field(..., description="Specific section reference (e.g., 31)")
    description: str = Field(..., description="Brief description of the standard's requirement")

class DocumentSegment(BaseModel):
    """Represents a segment of the document with metadata"""
    content: str = Field(..., description="The text content of the segment")
    page_number: Optional[int] = Field(None, description="Page number where the segment appears")
    
class AnalysisResult(BaseModel):
    """Structured analysis result with IFRS compliance details"""
    segment: DocumentSegment
    key_findings: list[str] = Field(..., description="Key observations from the analysis")
    compliance_status: Literal['compliant', 'non_compliant', 'needs_review'] = Field(..., description="Overall compliance status")
    relevant_standards: list[IFRSStandard] = Field(..., description="Relevant IFRS standards")
    recommendations: list[str] = Field(..., description="Specific recommendations for improvement")

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
        
        # Set default parameters
        from pydantic_ai.models.openai import ModelRequestParameters
        if parameters is None:
            parameters = ModelRequestParameters(
                temperature=0.7,
                max_tokens=2000,
                top_p=0.95
            )
        
        # Log request details
        logging.info(f"Sending request with messages: {messages}")
        logging.info(f"Settings: {settings}")
        logging.info(f"Parameters: {parameters}")
        
        # Make the request
        response = await super().request(messages, settings, parameters)
        
        # Log response details
        logging.info(f"Raw response: {response}")
        logging.info(f"Response type: {type(response)}")
        logging.info(f"Response dict: {response.__dict__ if hasattr(response, '__dict__') else 'No __dict__'}")
        
        return response
    
    def _process_response(self, response):
        # OpenRouter response format is different from OpenAI
        logging.info(f"Processing response type: {type(response)}")
        logging.info(f"Response attributes: {dir(response)}")
        
        try:
            # Set timestamp
            if not hasattr(response, 'created'):
                response.created = int(datetime.now().timestamp())
            
            # Process response data
            content = ''
            tool_calls = None
            finish_reason = 'stop'
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message'):
                    message = choice.message
                    if hasattr(message, 'content'):
                        content = message.content
                    if hasattr(message, 'tool_calls'):
                        tool_calls = message.tool_calls
                    if hasattr(choice, 'finish_reason'):
                        finish_reason = choice.finish_reason
            
            logging.info(f"Processed content: {content}")
            
            if not content and not tool_calls:
                # If no content or tool calls, try to extract from response object directly
                if hasattr(response, 'content'):
                    content = response.content
                elif hasattr(response, 'text'):
                    content = response.text
                elif hasattr(response, 'response'):
                    content = response.response
            
            # Create a properly formatted response
            from types import SimpleNamespace
            mock_response = SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        content=content or 'Please analyze the provided document for IFRS compliance.',
                        tool_calls=tool_calls
                    ),
                    finish_reason=finish_reason
                )],
                created=int(datetime.now().timestamp()),
                model=os.getenv('DEFAULT_LLM_MODEL')
            )
            
            return super()._process_response(mock_response)
        except Exception as e:
            logging.error(f"Error processing response: {e}")
            raise

# Initialize the document analyzer agent with OpenRouter
model = OpenRouterModel(
    os.getenv('DEFAULT_LLM_MODEL'),  # Model from .env
    base_url='https://openrouter.ai/api/v1',
    api_key=os.getenv('OPENROUTER_API_KEY')
)

document_analyzer = Agent(
    model,
    deps_type=AnalysisContext,  # Analysis context as dependencies
    result_type=list[AnalysisResult],  # Strongly typed analysis results
    system_prompt=(
        'You are an IFRS expert analyzing financial documents. '
        'Your task is to analyze document segments for compliance with IFRS standards.\n\n'
        'Focus on:\n'
        '- Revenue recognition principles (IFRS 15)\n'
        '- Financial instruments (IFRS 9)\n'
        '- Leasing arrangements (IFRS 16)\n'
        '- Impairment assessments (IAS 36)\n'
        '- Business combinations (IFRS 3)\n'
        '- Fair value measurements (IFRS 13)\n'
        '- Segment reporting (IFRS 8)\n\n'
        'Provide detailed analysis in markdown format.'
    ),
)

@document_analyzer.tool
async def read_document(ctx: RunContext[AnalysisContext]) -> list[DocumentSegment]:
    """Read document content based on file type"""
    document_path = ctx.deps.document_path
    if document_path.suffix.lower() == '.pdf':
        return read_pdf(document_path)
    else:
        with open(document_path, 'r') as f:
            content = f.read()
        return [DocumentSegment(page_number=1, content=content)]

def read_pdf(pdf_path: Path) -> list[DocumentSegment]:
    """Extract text from PDF with page numbers"""
    pages = []
    with open(pdf_path, 'rb') as file:
        pdf = pypdf.PdfReader(file)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text.strip():  # Only include non-empty pages
                pages.append(DocumentSegment(
                    page_number=i + 1,
                    content=text.strip()
                ))
    return pages

@document_analyzer.tool
async def segment_document(ctx: RunContext[AnalysisContext], pages: list[DocumentSegment]) -> list[DocumentSegment]:
    """Split document into logical sections based on content analysis"""
    segments = []
    current_content = []
    current_page = None
    min_segment_length = 500  # Minimum characters per segment
    
    for page in pages:
        lines = page.content.split('\n')
        page_num = page.page_number
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if not current_content:
                current_page = page_num
            
            current_content.append(line)
            current_text = '\n'.join(current_content)
            
            # Create a new segment when we have enough content
            if len(current_text) >= min_segment_length:
                segments.append(DocumentSegment(
                    content=current_text,
                    page_number=current_page
                ))
                current_content = []
    
    # Add the last section
    if current_content:
        segments.append(DocumentSegment(
            content='\n'.join(current_content),
            page_number=current_page
        ))
    
    return segments

@document_analyzer.tool
async def analyze_segment(ctx: RunContext[AnalysisContext], segment: DocumentSegment) -> AnalysisResult:
    """Analyze a single document segment and return structured analysis"""
    prompt = f"""Analyze this document segment for compliance with IFRS standards.
You must return a structured analysis that includes:
1. Key findings as a list of bullet points
2. Compliance status (one of: compliant, non_compliant, needs_review)
3. Relevant IFRS standards with section references
4. Specific recommendations as a list

Content from page {segment.page_number}:
{segment.content}

Format your response as a JSON object with the following structure:
{
    "key_findings": ["finding 1", "finding 2"],
    "compliance_status": "compliant|non_compliant|needs_review",
    "relevant_standards": [
        {"standard": "IFRS 15", "section": "31", "description": "Revenue recognition criteria"}
    ],
    "recommendations": ["recommendation 1", "recommendation 2"]
}"""

    # Get structured response from the model
    response = await ctx.model.complete_structured(prompt, AnalysisResult, 
        segment=segment,
        temperature=0.3  # Lower temperature for more consistent structured output
    )
    return response

class ReportGenerator:
    """Handles report generation from analysis results"""
    
    @staticmethod
    def generate_executive_summary(results: list[AnalysisResult]) -> str:
        """Generate an executive summary from the analysis results"""
        total_segments = len(results)
        compliance_stats = {
            'compliant': len([r for r in results if r.compliance_status == 'compliant']),
            'non_compliant': len([r for r in results if r.compliance_status == 'non_compliant']),
            'needs_review': len([r for r in results if r.compliance_status == 'needs_review'])
        }
        
        # Collect all unique standards
        all_standards = {}
        for result in results:
            for std in result.relevant_standards:
                key = f"{std.standard}.{std.section}"
                all_standards[key] = std
        
        summary = [
            "## Executive Summary\n",
            f"Total segments analyzed: {total_segments}",
            f"Compliance overview:",
            f"- Compliant sections: {compliance_stats['compliant']}",
            f"- Non-compliant sections: {compliance_stats['non_compliant']}",
            f"- Sections needing review: {compliance_stats['needs_review']}",
            "\n### Key IFRS Standards Referenced\n"
        ]
        
        for std in sorted(all_standards.values(), key=lambda x: x.standard):
            summary.append(f"- **{std.standard}.{std.section}**: {std.description}")
        
        return '\n'.join(summary)
    
    @staticmethod
    def generate_detailed_analysis(results: list[AnalysisResult]) -> str:
        """Generate detailed analysis section from results"""
        sections = ["\n## Detailed Analysis\n"]
        
        for result in results:
            sections.extend([
                f"### Analysis of Page {result.segment.page_number}\n",
                "#### Content Preview\n",
                f"`{result.segment.content[:200]}...`\n",
                "#### Key Findings\n",
                *[f"- {finding}" for finding in result.key_findings],
                "\n#### Compliance Status\n",
                f"**Status**: {result.compliance_status.replace('_', ' ').title()}\n",
                "#### Relevant IFRS Standards\n",
                *[f"- **{std.standard}.{std.section}**: {std.description}" 
                  for std in result.relevant_standards],
                "\n#### Recommendations\n",
                *[f"- {rec}" for rec in result.recommendations],
                "\n---\n"
            ])
        
        return '\n'.join(sections)

class DocumentAnalyzer:
    def __init__(self):
        """Initialize the document analyzer"""
        load_dotenv()
        
        # Model is already configured via the Agent initialization

    async def analyze_document(self, document_path: Path) -> str:
        """Analyze entire document and generate markdown report with streaming updates"""
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output') / 'analyses' / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create analysis context
        context = AnalysisContext(
            document_path=document_path,
            output_dir=output_dir
        )
        
        # Run the analysis
        logging.info(f"Starting analysis of {document_path}")
        result = await document_analyzer.run(
            'Analyze this financial document for IFRS compliance',
            deps=context
        )
        
        # Generate comprehensive report using structured data
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_parts = [
            "# IFRS Analysis Report",
            f"\nGenerated on: {timestamp}",
            f"\nDocument: {document_path}\n",
            ReportGenerator.generate_executive_summary(result.data),
            ReportGenerator.generate_detailed_analysis(result.data)
        ]
        
        # Combine report
        report = '\n'.join(report_parts)
        
        # Save report and analysis data
        report_path = output_dir / 'analysis.md'
        with open(report_path, 'w') as f:
            f.write(report)
            
        # Save structured analysis data as JSON for potential future use
        analysis_data_path = output_dir / 'analysis_data.json'
        with open(analysis_data_path, 'w') as f:
            json.dump(
                [result.dict() for result in result.data],
                f,
                indent=2,
                default=str  # Handle date/time objects
            )
        
        logging.info(f"Analysis completed. Report saved to {report_path}")
        logging.info(f"Structured analysis data saved to {analysis_data_path}")
        
        return report_path
