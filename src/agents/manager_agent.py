"""
Manager Agent Implementation for Chain of Agents
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

class ManagerAgent(BaseAgent):
    """
    Manager agent that synthesizes findings from worker agents
    Based on the Chain of Agents paper methodology
    """
    
    def __init__(
        self,
        name: str,
        capabilities: List[str],
        config: Optional[Dict] = None
    ):
        super().__init__(name, capabilities, config)
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
        
    async def _process_segment(
        self,
        input_cu: Optional[CommunicationUnit],
        context: ChainContext
    ) -> AgentResponse:
        """Synthesize findings from worker agents"""
        
        if not input_cu or not input_cu.accumulated_context:
            return AgentResponse(
                success=False,
                error="No accumulated context from workers"
            )
        
        # Extract all findings and evidence
        all_findings = input_cu.accumulated_context.get("findings", [])
        all_evidence = input_cu.evidence
        
        # Generate synthesis prompt
        prompt = self._generate_synthesis_prompt(
            all_findings,
            all_evidence,
            context
        )
        
        # Get LLM completion
        response = await self.llm.get_completion(
            prompt,
            temperature=0.3  # Keep synthesis focused and factual
        )
        
        # Parse synthesis response
        synthesis = self._parse_synthesis_response(response)
        
        return AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id=self._generate_cu_id(),
                type=CommunicationType.SYNTHESIS,
                source_agent=self.name,
                timestamp=self._get_timestamp(),
                segment_id="final_synthesis",
                segment_type="synthesis",
                segment_content="",  # No new content, just synthesis
                evidence=all_evidence,
                analysis=Analysis(
                    metrics=synthesis["metrics"],
                    findings=synthesis["key_findings"],
                    implications=synthesis["recommendations"]
                ),
                previous_findings=all_findings,
                accumulated_context=input_cu.accumulated_context
            )
        )
    
    def _generate_synthesis_prompt(
        self,
        findings: List[str],
        evidence: List[Evidence],
        context: ChainContext
    ) -> str:
        """Generate prompt for synthesizing findings"""
        
        # Format evidence for prompt
        evidence_text = "\n".join([
            f"- {e.content} [confidence: {e.confidence}]"
            for e in evidence
        ])
        
        prompt = f"""
        Task: Synthesize findings from IFRS document analysis
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
