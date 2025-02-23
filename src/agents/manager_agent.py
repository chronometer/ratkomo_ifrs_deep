"""
Manager Agent Implementation for Chain of Agents
"""
from typing import List, Dict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from ..models.openrouter_model import OpenRouterModel

class SynthesisInput(BaseModel):
    """Input for synthesis operation"""
    findings: List[str] = Field(..., description="List of findings from workers")
    evidence: List[Dict[str, Any]] = Field(..., description="Evidence supporting the findings")
    context: Dict[str, Any] = Field(..., description="Additional context for synthesis")

class SynthesisOutput(BaseModel):
    """Output from synthesis operation"""
    key_findings: List[str] = Field(..., description="Synthesized key findings")
    metrics: Dict[str, float] = Field(..., description="Quantitative metrics")
    recommendations: List[str] = Field(..., description="Strategic recommendations")
    confidence: float = Field(..., description="Overall confidence score")

class ManagerAgent(Agent):
    """
    Manager agent that synthesizes findings from worker agents
    Based on the Chain of Agents paper methodology
    """
    
    def __init__(
        self,
        model: OpenAIModel,
        name: str = "synthesis_manager",
        capabilities: Optional[List[str]] = None
    ):
        super().__init__(
            name=name,
            description="Synthesizes and analyzes findings from worker agents",
            capabilities=capabilities or ["synthesis", "analysis", "recommendation"]
        )
        self.model = model
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
        
    @Tool
    async def synthesize(
        self,
        input_data: SynthesisInput,
        context: RunContext
    ) -> SynthesisOutput:
        """Synthesize findings from worker agents"""
        # Generate synthesis prompt
        prompt = self._generate_synthesis_prompt(input_data)
        
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
        
        # Parse synthesis response
        return self._parse_synthesis_response(response)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for synthesis"""
        return """You are an expert synthesis manager for IFRS analysis.
        Your task is to:
        1. Synthesize findings from multiple analyses
        2. Identify key patterns and insights
        3. Calculate confidence scores
        4. Generate strategic recommendations
        5. Ensure consistency and clarity
        
        Focus on creating an executive-level synthesis while preserving important details."""
    
    def _generate_synthesis_prompt(self, input_data: SynthesisInput) -> str:
        """Generate prompt for synthesis"""
        # Format findings and evidence
        findings_text = "\n".join([f"- {f}" for f in input_data.findings])
        evidence_text = "\n".join([
            f"- {e['content']} [confidence: {e.get('confidence', 'N/A')}]"
            for e in input_data.evidence
        ])
        
        # Format context
        context_text = "\n".join([f"{k}: {v}" for k, v in input_data.context.items()])
        
        return f"""Synthesize these findings and evidence:

Findings:
{findings_text}

Evidence:
{evidence_text}

Context:
{context_text}

Provide:
1. Key findings synthesis
2. Metrics and measurements
3. Strategic recommendations
4. Confidence assessment"""

    def _parse_synthesis_response(self, response: str) -> SynthesisOutput:
        """Parse synthesis response with flexible format handling"""
        metrics = {}
        key_findings = []
        recommendations = []
        confidence = 0.0
        
        try:
            # Try JSON format first
            import json
            import re
            json_match = re.search(r'\{[^}]*"metrics":[^}]*\}', response)
            if json_match:
                data = json.loads(json_match.group(0))
                metrics = data.get('metrics', {})
                key_findings = data.get('key_findings', [])
                recommendations = data.get('recommendations', [])
                confidence = float(data.get('confidence', 0.7))
            else:
                # Try various text formats
                current_section = None
                
                # Support multiple section header formats:
                # METRICS: | Metrics: | ## Metrics | 2. Metrics
                section_pattern = re.compile(r'^(?:\d+\.)?\s*(?:#+\s*)?(\w+):', re.IGNORECASE)
                
                for line in response.split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check for section headers
                    section_match = section_pattern.match(line)
                    if section_match:
                        current_section = section_match.group(1).lower()
                        continue
                    
                    # Handle metrics in various formats:
                    # - accuracy: 0.95
                    # - accuracy = 0.95
                    # - accuracy → 0.95
                    elif current_section == 'metrics' and any(char in line for char in ':=→'):
                        try:
                            metric, value = re.split(r'[:=→]', line, 1)
                            metrics[metric.strip()] = float(value.strip())
                        except:
                            continue
                    
                    # Handle bullet points and numbered items
                    elif line.startswith(('- ', '* ', '• ', '→ ')) or re.match(r'^\d+\.\s', line):
                        content = re.sub(r'^(?:\d+\.)?\s*[-*•→]\s*', '', line).strip()
                        if current_section == 'key findings':
                            key_findings.append(content)
                        elif current_section == 'recommendations':
                            recommendations.append(content)
                    
                    # Handle confidence score in various formats
                    elif 'confidence' in line.lower() and any(char in line for char in ':=→'):
                        try:
                            confidence = float(re.split(r'[:=→]', line, 1)[1].strip())
                        except:
                            confidence = 0.7
        
        except Exception as e:
            logging.error(f"Error parsing synthesis response: {e}")
            # Provide basic structure even if parsing fails
            if not key_findings:
                key_findings = ["Error parsing synthesis response"]
            if not recommendations:
                recommendations = ["Review synthesis manually"]
            confidence = 0.5
        
        return SynthesisOutput(
            key_findings=key_findings,
            metrics=metrics,
            recommendations=recommendations,
            confidence=confidence
        )        Task: Synthesize findings from IFRS document analysis
        Standards in scope: {', '.join(context.standards_in_scope)}
        Fiscal period: {context.fiscal_period}
        Company: {context.company_info.get('name', 'Unknown')}
        
        All findings from document analysis:
        {self._format_findings(findings)}
        
        Supporting evidence:
        {evidence_text}
        
        Please synthesize the findings and provide:
        1. Key metrics across all sections
        2. Most important findings and their implications
        3. Recommendations for compliance and reporting
        4. Areas requiring attention or further analysis
        
        Format your response as:
        METRICS:
        - metric: value
        
        KEY FINDINGS:
        - finding 1
        - finding 2
        
        RECOMMENDATIONS:
        - recommendation 1
        - recommendation 2
        
        ATTENTION AREAS:
        - area 1
        - area 2
        """
        return prompt
    
    def _parse_synthesis_response(self, response: str) -> Dict[str, Any]:
        """Parse synthesis response into structured format"""
        synthesis = {
            "metrics": {},
            "key_findings": [],
            "recommendations": [],
            "attention_areas": []
        }
        
        current_section = ""
        for line in response.split("\n"):
            line = line.strip()
            if line in ["METRICS:", "KEY FINDINGS:", "RECOMMENDATIONS:", "ATTENTION AREAS:"]:
                current_section = line
            elif line.startswith("- "):
                content = line[2:].strip()
                if current_section == "METRICS:":
                    if ":" in content:
                        key, value = content.split(":", 1)
                        try:
                            synthesis["metrics"][key.strip()] = float(value.strip())
                        except ValueError:
                            synthesis["metrics"][key.strip()] = value.strip()
                elif current_section == "KEY FINDINGS:":
                    synthesis["key_findings"].append(content)
                elif current_section == "RECOMMENDATIONS:":
                    synthesis["recommendations"].append(content)
                elif current_section == "ATTENTION AREAS:":
                    synthesis["attention_areas"].append(content)
        
        return synthesis
    
    def _format_findings(self, findings: List[str]) -> str:
        """Format findings for synthesis prompt"""
        if not findings:
            return "No findings available"
            
        formatted = []
        for i, finding in enumerate(findings, 1):
            formatted.append(f"{i}. {finding}")
        return "\n".join(formatted)
