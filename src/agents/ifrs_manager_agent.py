"""
IFRS Manager Agent Implementation using pydantic-ai
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import re
import logging
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel
from ..models.ifrs_models import ComplianceAnalysis, FinalReport, IFRSStandard

class IFRSManagerAgent(Agent):
    """Manager agent that synthesizes IFRS analysis results"""
    
    def __init__(self, model: OpenAIModel):
        self.name = "ifrs_synthesis_manager"
        self.model = model
    
    async def synthesize_analyses(
        self,
        analyses: List[ComplianceAnalysis],
        context: RunContext
    ) -> FinalReport:
        """Synthesize multiple analyses into a final report"""
        
        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(analyses)
        
        # Get model completion
        response = await self.model.request(
            messages=[{
                "role": "system",
                "content": self._get_system_prompt()
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.3
        )
        
        # Parse response into final report
        return self._parse_synthesis_response(response, analyses)
    
    def _get_system_prompt(self) -> str:
        return """You are an expert IFRS synthesis manager. Your task is to analyze document segments for IFRS compliance.
        For each segment, provide:
        1. Key findings about IFRS compliance
        2. Overall compliance status (Compliant, Partially Compliant, Non-Compliant)
        3. Relevant IFRS standards with sections and descriptions
        4. Specific recommendations for improving compliance
        
        Format your response in a clear, structured way that can be parsed into sections.
        Use consistent formatting for each section:
        
        FINDINGS:
        - Finding 1
        - Finding 2
        
        COMPLIANCE STATUS:
        Overall: [status]
        Details: [explanation]
        
        STANDARDS:
        - Standard: [id]
          Section: [section]
          Description: [description]
        
        RECOMMENDATIONS:
        - Recommendation 1
        - Recommendation 2"""
    
    def _create_synthesis_prompt(self, analyses: List[ComplianceAnalysis]) -> str:
        # Combine all analyses into a structured prompt
        findings_summary = "\n".join([
            f"Analysis {i+1}:"
            f"\nType: {a.segment.segment_type}"
            f"\nFindings: {', '.join(a.key_findings)}"
            f"\nStandards: {', '.join(s.standard_id for s in a.standards)}"
            for i, a in enumerate(analyses)
        ])
        
        return f"""Synthesize these IFRS compliance analyses:

        {findings_summary}
        
        Provide:
        1. Overall compliance status
        2. Key findings across all analyses
        3. Risk areas requiring attention
        4. Strategic recommendations
        5. Standard-specific synthesis"""
    
    def _parse_synthesis_response(
        self,
        response: str,
        analyses: List[ComplianceAnalysis]
    ) -> FinalReport:
        """Parse synthesis response into final report"""
        
        # Generate markdown report
        report = []
        report.append("# IFRS Compliance Analysis Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Executive Summary
        report.append("## Executive Summary")
        report.append("Based on the analysis of the provided document segments, the following key points were identified:\n")

        # Key Findings
        report.append("## Key Findings")
        findings = self._extract_findings(response)
        for finding in findings:
            report.append(f"- {finding}")
        report.append("\n")

        # Compliance Status
        report.append("## Compliance Status")
        status = self._extract_compliance_status(response)
        report.append(f"Overall status: {status}\n")
        report.append("Detailed status by standard:\n")

        # Key Standards
        report.append("## Relevant IFRS Standards")
        standards = self._extract_standards(response)
        for standard in standards:
            report.append(f"### {standard.get('standard', '')}")
            if standard.get('section'):
                report.append(f"Section: {standard['section']}")
            report.append(f"Description: {standard.get('description', '')}\n")

        # Recommendations
        report.append("## Recommendations")
        recommendations = self._extract_recommendations(response)
        for recommendation in recommendations:
            report.append(f"- {recommendation}")
        report.append("\n")

        # Detailed Analysis
        report.append("## Detailed Analysis")
        for i, analysis in enumerate(analyses, 1):
            report.append(f"### Document Segment {i}\n")
            report.append("#### Content")
            report.append(f"```\n{analysis.segment.content}\n```\n")
            report.append("#### Analysis")
            report.append("##### Key Points")
            for finding in analysis.key_findings:
                report.append(f"- {finding}")
            report.append("\n##### Standards Referenced")
            for standard in analysis.standards:
                report.append(f"- {standard.standard_id}: {standard.description}")
            report.append("\n##### Recommendations")
            for recommendation in analysis.recommendations:
                report.append(f"- {recommendation}")
            report.append("\n")

        return FinalReport(
            content="\n".join(report),
            analyses=analyses,
            generated_at=datetime.now()
        )
        
    def _extract_findings(self, content: str) -> List[str]:
        """Extract key findings from analysis response"""
        findings = []
        in_findings = False
        
        for line in content.split('\n'):
            line = line.strip()
            if line.lower().startswith('findings:'):
                in_findings = True
                continue
            elif line.lower().startswith(('compliance status:', 'standards:', 'recommendations:')):
                in_findings = False
            elif in_findings and line.startswith('-'):
                findings.append(line[1:].strip())
        
        return findings
    
    def _extract_compliance_status(self, content: str) -> str:
        """Extract overall compliance status"""
        in_status = False
        
        for line in content.split('\n'):
            line = line.strip().lower()
            if line.startswith('compliance status:'):
                in_status = True
                continue
            elif in_status and line.startswith('overall:'):
                return line.split(':', 1)[1].strip()
            elif in_status and line.startswith(('standards:', 'recommendations:')):
                break
        
        return 'Unknown'
    
    def _extract_standards(self, content: str) -> List[Dict[str, str]]:
        """Extract relevant IFRS standards"""
        standards = []
        current_standard = {}
        in_standards = False
        
        for line in content.split('\n'):
            line = line.strip()
            if line.lower().startswith('standards:'):
                in_standards = True
                continue
            elif line.lower().startswith('recommendations:'):
                in_standards = False
            elif in_standards:
                if line.startswith('- Standard:'):
                    if current_standard:
                        standards.append(current_standard)
                    current_standard = {'standard': line.split(':', 1)[1].strip()}
                elif line.startswith('Section:'):
                    current_standard['section'] = line.split(':', 1)[1].strip()
                elif line.startswith('Description:'):
                    current_standard['description'] = line.split(':', 1)[1].strip()
        
        if current_standard:
            standards.append(current_standard)
        
        return standards
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from analysis"""
        recommendations = []
        in_recommendations = False
        
        for line in content.split('\n'):
            line = line.strip()
            if line.lower().startswith('recommendations:'):
                in_recommendations = True
                continue
            elif in_recommendations and line.startswith('-'):
                recommendations.append(line[1:].strip())
        
        return recommendations
        
        # Combine standards analysis
        standards_map = {}
        for analysis in analyses:
            for standard in analysis.standards:
                if standard.standard_id not in standards_map:
                    standards_map[standard.standard_id] = standard
                else:
                    # Merge findings and evidence
                    existing = standards_map[standard.standard_id]
                    existing.findings.extend(standard.findings)
                    existing.evidence.extend(standard.evidence)
        
        return FinalReport(
            overall_compliance=overall_compliance,
            key_findings=key_findings,
            standards_analysis=list(standards_map.values()),
            recommendations=recommendations,
            risk_areas=risk_areas,
            timestamp=datetime.now().isoformat()
        )
