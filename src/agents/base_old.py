"""
Base agent implementation for Chain of Agents
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from src.config.llm_config import LLMManager, get_llm_config
from src.config.tracing_config import TracingManager, TracingConfig
from .communication import (
    CommunicationUnit, 
    ChainContext, 
    AgentResponse,
    CommunicationType,
    Evidence
)

class AgentCapability:
    """Defines what an agent can do"""
    def __init__(
        self,
        name: str,
        description: str,
        required_inputs: List[str],
        produced_outputs: List[str]
    ):
        self.name = name
        self.description = description
        self.required_inputs = required_inputs
        self.produced_outputs = produced_outputs

class BaseAgent(ABC):
    """Base class for all agents in the chain"""
    
    def __init__(
        self,
        name: str,
        capabilities: List[AgentCapability],
        config: Optional[Dict] = None
    ):
        self.name = name
        self.capabilities = capabilities
        self.config = config or {}
        
        # Initialize LLM and tracing
        self.llm = LLMManager(get_llm_config())
        self.tracing = TracingManager(TracingConfig())
        
        # Chain configuration
        self.next_agent: Optional[BaseAgent] = None
        
    def _generate_cu_id(self) -> str:
        """Generate unique ID for communication unit"""
        return str(uuid.uuid4())
        
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        return datetime.now().isoformat()
        self.previous_agent: Optional[BaseAgent] = None
        
    async def process(
        self,
        input_cu: Optional[CommunicationUnit],
        context: ChainContext
    ) -> AgentResponse:
        """
        Process input and generate response
        
        Args:
            input_cu: Input communication unit from previous agent
            context: Current chain context
            
        Returns:
            AgentResponse with success status and new communication unit
        """
        try:
            with self.tracing.trace_operation(
                agent_name=self.name,
                operation="process",
                metadata={"context_id": context.document_id}
            ) as trace_id:
                # Validate input
                if not self._validate_input(input_cu):
                    return AgentResponse(
                        success=False,
                        error="Invalid input communication unit"
                    )
                
                # Process current segment
                processing_result = await self._process_segment(input_cu, context)
                if not processing_result.success:
                    return processing_result
                
                # Generate communication unit
                cu = await self._generate_communication_unit(
                    input_cu,
                    processing_result,
                    context
                )
                
                # Validate output
                if not self._validate_output(cu):
                    return AgentResponse(
                        success=False,
                        error="Generated invalid communication unit"
                    )
                
                return AgentResponse(
                    success=True,
                    communication_unit=cu,
                    metrics=processing_result.metrics
                )
                
        except Exception as e:
            self.tracing.log_event(
                trace_id,
                "error",
                {"error": str(e)}
            )
            return AgentResponse(
                success=False,
                error=f"Processing error: {str(e)}"
            )
    
    @abstractmethod
    async def _process_segment(
        self,
        input_cu: Optional[CommunicationUnit],
        context: ChainContext
    ) -> AgentResponse:
        """
        Process the current segment
        
        Args:
            input_cu: Input communication unit
            context: Chain context
            
        Returns:
            Processing result
        """
        pass
    
    async def _generate_communication_unit(
        self,
        input_cu: Optional[CommunicationUnit],
        processing_result: AgentResponse,
        context: ChainContext
    ) -> CommunicationUnit:
        """
        Generate communication unit for next agent
        
        Args:
            input_cu: Input communication unit
            processing_result: Result of processing
            context: Chain context
            
        Returns:
            New communication unit
        """
        # Get previous findings if any
        previous_findings = []
        if input_cu:
            previous_findings.extend(input_cu.previous_findings)
            if input_cu.evidence:
                previous_findings.append({
                    "agent": input_cu.source_agent,
                    "evidence": input_cu.evidence
                })
        
        # Create new communication unit
        if processing_result.communication_unit:
            # Use values from processing result
            return CommunicationUnit(
                id=f"cu_{uuid.uuid4().hex[:8]}",
                type=CommunicationType.EVIDENCE,
                source_agent=self.name,
                target_agent=self.next_agent.name if self.next_agent else None,
                timestamp=datetime.utcnow().isoformat(),
                segment_id=processing_result.communication_unit.segment_id,
                segment_type=processing_result.communication_unit.segment_type,
                segment_content=processing_result.communication_unit.segment_content,
                evidence=processing_result.communication_unit.evidence,
                analysis=processing_result.communication_unit.analysis,
                previous_findings=previous_findings,
                accumulated_context=self._update_context(
                    input_cu.accumulated_context if input_cu else {},
                    processing_result
                ),
                metadata={
                    "agent_version": "1.0",
                    "processing_time": processing_result.metrics.get("processing_time")
                }
            )
        else:
            # Fallback to default values
            return CommunicationUnit(
                id=f"cu_{uuid.uuid4().hex[:8]}",
                type=CommunicationType.EVIDENCE,
                source_agent=self.name,
                target_agent=self.next_agent.name if self.next_agent else None,
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_001",  # Default segment ID that matches pattern
                segment_type="financial_statement",
                segment_content="No content",  # Non-empty default content
                evidence=[],
                previous_findings=previous_findings,
                accumulated_context=self._update_context(
                    input_cu.accumulated_context if input_cu else {},
                    processing_result
                ),
                metadata={
                    "agent_version": "1.0",
                    "processing_time": processing_result.metrics.get("processing_time")
                }
            )
    
    def _validate_input(self, input_cu: Optional[CommunicationUnit]) -> bool:
        """
        Validate input communication unit
        
        Args:
            input_cu: Input communication unit
            
        Returns:
            True if valid, False otherwise
        """
        if not input_cu:
            return True  # First agent in chain
            
        # Check required fields
        if not input_cu.id or not input_cu.source_agent:
            return False
            
        # Validate previous agent
        if self.previous_agent and input_cu.source_agent != self.previous_agent.name:
            return False
            
        return True
    
    def _validate_output(self, cu: CommunicationUnit) -> bool:
        """
        Validate output communication unit
        
        Args:
            cu: Generated communication unit
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not cu.id or not cu.source_agent:
            return False
            
        # Validate evidence
        if cu.type == CommunicationType.EVIDENCE and not cu.evidence:
            return False
            
        # Validate analysis
        if cu.type == CommunicationType.ANALYSIS and not cu.analysis:
            return False
            
        return True
    
    def _update_context(
        self,
        current_context: Dict[str, Any],
        processing_result: AgentResponse
    ) -> Dict[str, Any]:
        """
        Update accumulated context with new information
        
        Args:
            current_context: Current accumulated context
            processing_result: Result of processing
            
        Returns:
            Updated context
        """
        # Start with current context
        updated_context = current_context.copy()
        
        # Add new context from processing result
        if processing_result.communication_unit:
            cu = processing_result.communication_unit
            if cu.evidence:
                updated_context["latest_evidence"] = [
                    e.dict() for e in cu.evidence
                ]
            if cu.analysis:
                updated_context["latest_analysis"] = cu.analysis.dict()
        
        return updated_context

# Example usage:
"""
class DocumentAnalyzer(BaseAgent):
    async def _process_segment(
        self,
        input_cu: Optional[CommunicationUnit],
        context: ChainContext
    ) -> AgentResponse:
        # Process document segment
        prompt = f\"\"\"
        Analyze this financial document segment:
        {input_cu.segment_content if input_cu else context.current_segment}
        
        Previous findings:
        {input_cu.previous_findings if input_cu else []}
        
        Identify:
        1. Key financial metrics
        2. IFRS compliance issues
        3. Required disclosures
        \"\"\"
        
        response = await self.llm.get_completion(prompt)
        
        # Create evidence
        evidence = [
            Evidence(
                source_location="segment_1",
                content="Found compliance issue with IAS 1",
                confidence=0.95
            )
        ]
        
        return AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="cu_001",
                type=CommunicationType.EVIDENCE,
                source_agent=self.name,
                evidence=evidence
            )
        )
"""
