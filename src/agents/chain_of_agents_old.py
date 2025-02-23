"""Chain of Agents (CoA) Implementation for IFRS Document Analysis
Based on the Chain of Agents paper methodology
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from src.config.llm_config import LLMManager, get_llm_config
from src.config.tracing_config import TracingManager, TracingConfig
from .base import BaseAgent
from .worker_agent import WorkerAgent
from .manager_agent import ManagerAgent
from .communication import (
    CommunicationUnit,
    ChainContext,
    AgentResponse,
    CommunicationType,
    Evidence
)

class ChainOfAgents:
    """
    Chain of Agents implementation for IFRS document analysis
    Based on the Chain of Agents paper methodology
    """
    
    def __init__(
        self,
        num_workers: int = 3,
        segment_size: int = 1000,
        config: Optional[Dict] = None
    ):
        self.config = config or {}
        self.num_workers = num_workers
        self.segment_size = segment_size
        
        # Initialize agents
        self.workers = [
            WorkerAgent(
                name=f"worker_{i}",
                capabilities=["document_analysis", "ifrs_compliance"],
                segment_size=segment_size,
                config=config
            )
            for i in range(num_workers)
        ]
        
        self.manager = ManagerAgent(
            name="synthesis_manager",
            capabilities=["synthesis", "recommendation"],
            config=config
        )
        
        # Connect workers in chain
        for i in range(num_workers - 1):
            self.workers[i].next_agent = self.workers[i + 1]
        self.workers[-1].next_agent = self.manager
        
        # Initialize tracing
        self.tracing = TracingManager(TracingConfig())
        
    async def process_document(
        self,
        document: Dict[str, Any],
        task_type: str,
        standards: List[str],
        fiscal_period: Dict[str, str],
        company_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process document through chain of agents
        
        Args:
            document: Document to analyze
            task_type: Type of analysis task
            standards: IFRS standards to check
            fiscal_period: Fiscal period details
            company_info: Company information
            
        Returns:
            Dict containing analysis results
        """
        try:
            with self.tracing.trace_operation(
                agent_name="chain_of_agents",
                operation="process_document",
                metadata={
                    "document_id": document.get("id"),
                    "task_type": task_type
                }
            ) as trace_id:
                # Create chain context
                doc_id = document.get("id")
                if not doc_id or not doc_id.startswith("doc_"):
                    doc_id = f"doc_{str(uuid.uuid4())[:8]}"
                
                context = ChainContext(
                    document_id=doc_id,
                    task_type=task_type,
                    start_time=datetime.utcnow().isoformat(),
                    total_segments=self._estimate_segments(document),
                    standards_in_scope=standards,
                    fiscal_period=fiscal_period,
                    company_info=company_info
                )
                
                # Segment document
                segments = await self._segment_document(document)
                
                # Handle empty document
                if not segments:
                    # Create empty segment for processing
                    empty_segment = {
                        "id": "seg_0",
                        "content": "Empty document",
                        "start_pos": 0,
                        "end_pos": 0
                    }
                    initial_cu = await self._create_initial_cu(empty_segment)
                    current_response = await self.workers[0].process(initial_cu, context)
                    
                    if not current_response.success:
                        return {
                            "success": False,
                            "error": current_response.error
                        }
                else:
                    # Process first segment with first worker
                    initial_cu = await self._create_initial_cu(segments[0])
                    current_response = await self.workers[0].process(initial_cu, context)
                    
                    if not current_response.success:
                        return {
                            "success": False,
                            "error": current_response.error
                        }
                
                # Process remaining segments
                for segment in segments[1:]:
                    # Update context with new segment
                    context.segments_processed += 1
                    
                    # Create new communication unit for segment
                    segment_cu = self._create_segment_cu(
                        segment,
                        current_response.communication_unit
                    )
                    
                    # Process through first worker
                    current_response = await self.workers[0].process(
                        segment_cu,
                        context
                    )
                    
                    if not current_response.success:
                        return {
                            "success": False,
                            "error": current_response.error
                        }
                
                # Get final synthesis from manager
                final_response = await self.manager.process(
                    current_response.communication_unit,
                    context
                )
                
                if not final_response.success:
                    return {
                        "success": False,
                        "error": final_response.error
                    }
                
                # Extract results from final synthesis
                results = self._extract_results(final_response.communication_unit)
                
                return {
                    "success": True,
                    "results": results,
                    "metrics": final_response.metrics
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Chain processing failed: {str(e)}"
            }
    
    async def _segment_document(self, document: Dict) -> List[Dict]:
        """Split document into segments"""
        # This could be enhanced with more sophisticated segmentation
        content = document.get("content", "")
        segments = []
        
        # Simple length-based segmentation
        for i in range(0, len(content), self.segment_size):
            segment_content = content[i:i + self.segment_size]
            segments.append({
                "id": f"seg_{len(segments)}",
                "content": segment_content,
                "start_pos": i,
                "end_pos": i + len(segment_content)
            })
        
        return segments
    
    def _estimate_segments(self, document: Dict) -> int:
        """Estimate number of segments"""
        content_length = len(document.get("content", ""))
        # Always return at least 1 segment, even for empty documents
        return max(1, (content_length + self.segment_size - 1) // self.segment_size)
    
    async def _create_initial_cu(
        self,
        segment: Dict
    ) -> CommunicationUnit:
        """Create initial communication unit"""
        return CommunicationUnit(
            id=str(uuid.uuid4()),
            type=CommunicationType.EVIDENCE,
            source_agent="chain_orchestrator",
            timestamp=datetime.utcnow().isoformat(),
            segment_id=segment["id"],
            segment_type="document_segment",
            segment_content=segment["content"],
            evidence=[],
            previous_findings=[],
            accumulated_context={}
        )
    
    def _create_segment_cu(
        self,
        segment: Dict,
        previous_cu: CommunicationUnit
    ) -> CommunicationUnit:
        """Create communication unit for new segment"""
        return CommunicationUnit(
            id=str(uuid.uuid4()),
            type=CommunicationType.EVIDENCE,
            source_agent="chain_orchestrator",
            timestamp=datetime.utcnow().isoformat(),
            segment_id=segment["id"],
            segment_type="document_segment",
            segment_content=segment["content"],
            evidence=previous_cu.evidence,
            previous_findings=previous_cu.previous_findings,
            accumulated_context=previous_cu.accumulated_context
        )
    
    def _extract_results(self, final_cu: CommunicationUnit) -> Dict[str, Any]:
        """Extract final results from manager's synthesis"""
        return {
            "metrics": final_cu.analysis.metrics if final_cu.analysis else {},
            "findings": final_cu.analysis.findings if final_cu.analysis else [],
            "implications": final_cu.analysis.implications if final_cu.analysis else [],
            "evidence": [
                {
                    "content": e.content,
                    "location": e.source_location,
                    "confidence": e.confidence
                }
                for e in final_cu.evidence
            ] if final_cu.evidence else []
        }
