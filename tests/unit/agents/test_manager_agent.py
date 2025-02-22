"""Unit tests for ManagerAgent class"""
import pytest
from datetime import datetime
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.manager_agent import ManagerAgent
from src.agents.communication import (
    CommunicationUnit,
    ChainContext,
    AgentResponse,
    CommunicationType,
    Evidence,
    Analysis
)

@pytest.fixture
def mock_llm_response():
    """Mock LLM response fixture"""
    return """
    METRICS:
    - total_revenue: 5000000
    - overall_compliance: 85.5
    
    KEY FINDINGS:
    - IFRS 15 compliance is satisfactory across all segments
    - IFRS 16 requires attention in lease disclosure areas
    
    RECOMMENDATIONS:
    - Maintain current revenue recognition practices
    - Enhance lease disclosure documentation
    
    ATTENTION AREAS:
    - Lease contract documentation
    - Related party disclosures
    """

@pytest.fixture
def manager_agent():
    """Fixture for manager agent"""
    agent = ManagerAgent(
        name="test_manager",
        capabilities=["synthesis", "recommendation"]
    )
    # Mock LLM for testing
    agent.llm.get_completion = AsyncMock(return_value=mock_llm_response())
    return agent

@pytest.fixture
def test_context():
    """Fixture for chain context"""
    return ChainContext(
        document_id="test_doc",
        task_type="ifrs_compliance",
        start_time=datetime.utcnow().isoformat(),
        total_segments=3,
        standards_in_scope=["IFRS 15", "IFRS 16"],
        fiscal_period={
            "start": "2024-01-01",
            "end": "2024-12-31"
        },
        company_info={
            "name": "Test Corp",
            "industry": "Technology"
        }
    )

@pytest.fixture
def test_accumulated_findings():
    """Fixture for accumulated findings from workers"""
    return [
        "Revenue recognition follows IFRS 15 guidelines",
        "Contract assets properly classified",
        "Lease terms need better documentation",
        "Performance obligations clearly identified"
    ]

@pytest.fixture
def test_evidence():
    """Fixture for evidence from workers"""
    return [
        Evidence(
            source_location="section 2.1",
            content="Revenue recognition evidence",
            confidence=0.95
        ),
        Evidence(
            source_location="section 3.4",
            content="Lease documentation evidence",
            confidence=0.85
        )
    ]

@pytest.fixture
def test_communication_unit(test_accumulated_findings, test_evidence):
    """Fixture for input communication unit"""
    return CommunicationUnit(
        id="test_cu",
        type=CommunicationType.EVIDENCE,
        source_agent="worker_agent",
        timestamp=datetime.utcnow().isoformat(),
        segment_id="final_segment",
        segment_type="financial_statement",
        segment_content="Final segment content",
        evidence=test_evidence,
        previous_findings=test_accumulated_findings,
        accumulated_context={
            "findings": test_accumulated_findings
        }
    )

@pytest.mark.unit
class TestManagerAgent:
    """Test cases for ManagerAgent class"""
    
    def test_manager_initialization(self, manager_agent):
        """Test manager agent initialization"""
        assert manager_agent.name == "test_manager"
        assert "synthesis" in manager_agent.capabilities
        assert "recommendation" in manager_agent.capabilities
    
    @pytest.mark.asyncio
    async def test_process_segment(
        self,
        manager_agent,
        test_context,
        test_communication_unit,
        mock_llm_response
    ):
        """Test synthesis processing"""
        response = await manager_agent._process_segment(
            test_communication_unit,
            test_context
        )
        
        assert response.success
        assert response.communication_unit is not None
        assert response.communication_unit.type == CommunicationType.SYNTHESIS
        assert response.communication_unit.analysis is not None
        assert len(response.communication_unit.analysis.findings) > 0
    
    def test_parse_synthesis_response(self, manager_agent, mock_llm_response):
        """Test synthesis response parsing"""
        synthesis = manager_agent._parse_synthesis_response(mock_llm_response)
        
        assert "metrics" in synthesis
        assert "key_findings" in synthesis
        assert "recommendations" in synthesis
        assert "attention_areas" in synthesis
        
        assert synthesis["metrics"]["total_revenue"] == 5000000
        assert len(synthesis["key_findings"]) == 2
        assert len(synthesis["recommendations"]) == 2
        assert len(synthesis["attention_areas"]) == 2
    
    @pytest.mark.asyncio
    async def test_generate_synthesis_prompt(
        self,
        manager_agent,
        test_context,
        test_accumulated_findings,
        test_evidence
    ):
        """Test synthesis prompt generation"""
        prompt = manager_agent._generate_synthesis_prompt(
            test_accumulated_findings,
            test_evidence,
            test_context
        )
        
        assert "IFRS 15" in prompt
        assert "IFRS 16" in prompt
        assert "Test Corp" in prompt
        assert "Technology" in prompt
        assert "Revenue recognition" in prompt
    
    def test_format_findings(self, manager_agent, test_accumulated_findings):
        """Test findings formatting"""
        formatted = manager_agent._format_findings(test_accumulated_findings)
        
        assert all(str(i) in formatted for i in range(1, len(test_accumulated_findings) + 1))
        assert "Revenue recognition" in formatted
        assert "Lease terms" in formatted
    
    @pytest.mark.asyncio
    async def test_process_without_findings(
        self,
        manager_agent,
        test_context
    ):
        """Test processing without accumulated findings"""
        empty_cu = CommunicationUnit(
            id="empty_cu",
            type=CommunicationType.EVIDENCE,
            source_agent="worker_agent",
            evidence=[],
            accumulated_context={}
        )
        
        response = await manager_agent.process(empty_cu, test_context)
        
        assert not response.success
        assert "No accumulated context" in response.error
    
    @pytest.mark.asyncio
    async def test_process_with_error_handling(
        self,
        manager_agent,
        test_context,
        test_communication_unit
    ):
        """Test error handling during processing"""
        # Mock LLM to raise an exception
        manager_agent.llm.get_completion = AsyncMock(
            side_effect=Exception("LLM error")
        )
        
        response = await manager_agent.process(
            test_communication_unit,
            test_context
        )
        
        assert not response.success
        assert "LLM error" in response.error
    
    @pytest.mark.asyncio
    async def test_full_synthesis_flow(
        self,
        manager_agent,
        test_context,
        test_communication_unit
    ):
        """Test complete synthesis flow"""
        # Add some metrics to test context
        test_communication_unit.accumulated_context["metrics"] = {
            "revenue": 1000000,
            "compliance_score": 85
        }
        
        response = await manager_agent.process(
            test_communication_unit,
            test_context
        )
        
        assert response.success
        cu = response.communication_unit
        assert cu.type == CommunicationType.SYNTHESIS
        assert cu.analysis is not None
        assert cu.analysis.metrics is not None
        assert len(cu.analysis.findings) > 0
        assert len(cu.analysis.implications) > 0
        assert all(evidence.confidence > 0 for evidence in cu.evidence)
    
    def test_synthesis_prioritization(self, manager_agent, mock_llm_response):
        """Test findings prioritization in synthesis"""
        synthesis = manager_agent._parse_synthesis_response(mock_llm_response)
        
        # Verify attention areas are properly identified
        attention_areas = synthesis["attention_areas"]
        assert len(attention_areas) > 0
        assert "Lease contract documentation" in attention_areas
        
        # Verify recommendations are properly ordered
        recommendations = synthesis["recommendations"]
        assert len(recommendations) > 0
        assert any("enhance" in r.lower() for r in recommendations)
