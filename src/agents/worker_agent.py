"""
Worker Agent Implementation for Chain of Agents
"""
from typing import List, Dict, Any, Optional
from .base import BaseAgent
from .communication import (
    CommunicationUnit,
    ChainContext,
    AgentResponse,
    CommunicationType,
    Evidence,
    Analysis
)

class WorkerAgent(BaseAgent):
    """
    Worker agent that processes document segments and communicates with other workers
    Based on the Chain of Agents paper methodology
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[str],
        segment_size: int = 1000,
        config: Optional[Dict] = None
    ):
        super().__init__(name, capabilities, config)
        self.segment_size = segment_size
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
        
    async def _process_segment(
        self,
        input_cu: Optional[CommunicationUnit],
        context: ChainContext
    ) -> AgentResponse:
        """Process current segment and previous findings"""
        
        # Extract current segment and previous context
        current_segment = input_cu.segment_content if input_cu else ""
        previous_findings = input_cu.previous_findings if input_cu else []
        
        # Generate prompt incorporating previous findings
        prompt = self._generate_segment_prompt(
            current_segment,
            previous_findings,
            context
        )
        
        # Get LLM completion
        response = await self.llm.get_completion(
            prompt,
            temperature=0.7  # Allow some creativity in analysis
        )
        
        # Extract findings and evidence
        findings, evidence = self._parse_llm_response(response)
        
        # Create analysis object
        analysis = Analysis(
            metrics=self._extract_metrics(findings),
            findings=findings,
            implications=self._derive_implications(findings)
        )
        
        return AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id=self._generate_cu_id(),
                type=CommunicationType.ANALYSIS,
                source_agent=self.name,
                timestamp=self._get_timestamp(),
                segment_id=input_cu.segment_id if input_cu else "initial",
                segment_type=input_cu.segment_type if input_cu else "unknown",
                segment_content=current_segment,
                evidence=evidence,
                analysis=analysis,
                previous_findings=previous_findings,
                accumulated_context=self._update_context(
                    input_cu.accumulated_context if input_cu else {},
                    findings
                )
            )
        )
    
    def _generate_segment_prompt(
        self,
        current_segment: str,
        previous_findings: List[Dict],
        context: ChainContext
    ) -> str:
        """Generate prompt for LLM incorporating segment and context"""
        
        # Start with task context
        prompt = f"""
        Task: Analyze this financial document segment according to IFRS standards
        Standards in scope: {', '.join(context.standards_in_scope)}
        Fiscal period: {context.fiscal_period}
        
        Previous findings:
        {self._format_previous_findings(previous_findings)}
        
        Current segment:
        {current_segment}
        
        Please analyze this segment and provide:
        1. Key financial metrics and their values
        2. Compliance findings related to IFRS standards
        3. Supporting evidence from the text
        4. Implications of the findings
        
        Format your response as:
        METRICS:
        - metric: value
        
        FINDINGS:
        - finding 1
        - finding 2
        
        EVIDENCE:
        - evidence 1 [location] [confidence]
        - evidence 2 [location] [confidence]
        
        IMPLICATIONS:
        - implication 1
        - implication 2
        """
        return prompt
    
    def _parse_llm_response(self, response: str) -> tuple[List[str], List[Evidence]]:
        """Parse LLM response into findings and evidence"""
        findings = []
        evidence_list = []
        
        # Simple parsing - could be enhanced with regex
        current_section = ""
        for line in response.split("\n"):
            line = line.strip()
            if line in ["METRICS:", "FINDINGS:", "EVIDENCE:", "IMPLICATIONS:"]:
                current_section = line
            elif line.startswith("- "):
                if current_section == "FINDINGS:":
                    findings.append(line[2:])
                elif current_section == "EVIDENCE:":
                    # Parse evidence with location and confidence
                    parts = line[2:].split("[")
                    content = parts[0].strip()
                    location = parts[1].strip("]") if len(parts) > 1 else "unknown"
                    confidence = float(parts[2].strip("]")) if len(parts) > 2 else 0.8
                    
                    evidence_list.append(Evidence(
                        source_location=location,
                        content=content,
                        confidence=confidence,
                        context={}
                    ))
        
        return findings, evidence_list
    
    def _extract_metrics(self, findings: List[str]) -> Dict[str, Any]:
        """Extract financial metrics from findings"""
        metrics = {}
        for finding in findings:
            if ":" in finding:
                key, value = finding.split(":", 1)
                try:
                    # Try to convert to number if possible
                    metrics[key.strip()] = float(value.strip())
                except ValueError:
                    metrics[key.strip()] = value.strip()
        return metrics
    
    def _derive_implications(self, findings: List[str]) -> List[str]:
        """Derive implications from findings"""
        # This could be enhanced with another LLM call
        return [f"Implication of {finding}" for finding in findings]
    
    def _format_previous_findings(self, findings: List[Dict]) -> str:
        """Format previous findings for prompt"""
        if not findings:
            return "No previous findings"
            
        formatted = []
        for i, finding in enumerate(findings, 1):
            formatted.append(f"{i}. {finding.get('content', '')}")
        return "\n".join(formatted)
    
    def _update_context(
        self,
        current_context: Dict[str, Any],
        new_findings: List[str]
    ) -> Dict[str, Any]:
        """Update accumulated context with new findings"""
        context = current_context.copy()
        
        # Add new findings to context
        if "findings" not in context:
            context["findings"] = []
        context["findings"].extend(new_findings)
        
        # Could add more sophisticated context updates here
        
        return context
