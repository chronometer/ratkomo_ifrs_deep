"""
Basic document analyzer for IFRS documents
"""
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel

class DocumentSegment(BaseModel):
    """Represents a segment of the document"""
    content: str
    title: Optional[str] = None

class DocumentAnalyzer:
    def __init__(self):
        """Initialize the document analyzer"""
        load_dotenv()
        
        self.llm = ChatOpenAI(
            model=os.getenv('DEFAULT_LLM_MODEL'),
            temperature=0.3,
            openai_api_key=os.getenv('OPENROUTER_API_KEY'),
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=1000,
            default_headers={
                "HTTP-Referer": "https://github.com/jargothia/Agent_chain_doc_analysis",
                "X-Title": "IFRS Analysis Agent"
            }
        )
        
    def segment_document(self, content: str) -> List[DocumentSegment]:
        """Split document into logical sections based on titles"""
        segments = []
        lines = content.split('\n')
        
        current_title = None
        current_content = []
        
        for line in lines:
            # Detect section titles (all caps, not too long)
            if line.strip().isupper() and len(line.strip()) < 100:
                # Save previous section if exists
                if current_content:
                    segments.append(DocumentSegment(
                        content='\n'.join(current_content),
                        title=current_title
                    ))
                current_title = line.strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            segments.append(DocumentSegment(
                content='\n'.join(current_content),
                title=current_title
            ))
        
        return segments

    async def analyze_segment(self, segment: DocumentSegment) -> str:
        """Analyze a single document segment and return markdown analysis"""
        prompt = f"""You are an IFRS expert. Analyze this document segment for compliance with IFRS standards.

Section: {segment.title or 'Untitled Section'}

Content:
{segment.content}

Provide your analysis in markdown format focusing on:
1. Key findings and observations
2. Compliance assessment
3. Specific recommendations
4. Relevant IFRS standards

Use proper markdown formatting with headers (##), bullet points (-), and emphasis (**) where appropriate.
Be specific about IFRS standard references (e.g., 'IFRS 15.31' instead of just 'IFRS 15')."""

        messages = [
            SystemMessage(content="You are an IFRS expert analyzing financial documents. Provide clear, concise analysis in markdown format."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        return response.content

    async def analyze_document(self, document_path: Path) -> str:
        """Analyze entire document and generate markdown report"""
        # Read document
        with open(document_path, 'r') as f:
            content = f.read()
        
        # Segment document
        segments = self.segment_document(content)
        
        # Generate report header
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_parts = [
            "# IFRS Analysis Report",
            f"\nGenerated on: {timestamp}",
            f"\nDocument: {document_path}\n",
            "## Executive Summary",
            "\nThis report provides a detailed analysis of the document's compliance with IFRS standards.\n",
            "## Detailed Analysis"
        ]
        
        # Analyze each segment
        for segment in segments:
            if segment.title:
                report_parts.append(f"\n### {segment.title}")
            analysis = await self.analyze_segment(segment)
            report_parts.append(analysis)
        
        # Combine report
        report = '\n'.join(report_parts)
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path('output') / f'analysis_{timestamp}.md'
        with open(output_path, 'w') as f:
            f.write(report)
        
        return output_path
