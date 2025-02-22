"""Unit tests for BaseAgent class"""
import pytest
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import ValidationError

from src.agents.base import BaseAgent, AgentCapability
from src.agents.communication import (
    CommunicationUnit,
    ChainContext,
    AgentResponse,
    CommunicationType,
    Evidence
)

class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for testing"""
    async def _process_segment(
        self,
        input_cu: Optional[CommunicationUnit],
        context: ChainContext
    ) -> AgentResponse:
        """Simple implementation for testing"""
        try:
            cu = CommunicationUnit(
                id="cu_test_123",
                type=CommunicationType.EVIDENCE,
                source_agent=self.name,
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_123",  # Matches pattern ^seg_[0-9]+$
                segment_type="test_type",
                segment_content="test_content",
                evidence=[
                    Evidence(
                        source_location="test_loc",
                        content="test evidence",
                        confidence=0.9
                    )
                ]
            )
            if self.next_agent:
                cu.target_agent = self.next_agent.name
            return AgentResponse(
                success=True,
                communication_unit=cu
            )
        except Exception as e:
            return AgentResponse(
                success=False,
                error=str(e)
            )

@pytest.fixture
def test_agent():
    """Fixture for test agent"""
    capabilities = [
        AgentCapability(
            name="test_capability",
            description="Test capability",
            required_inputs=["test_input"],
            produced_outputs=["test_output"]
        )
    ]
    return TestAgent(
        name="test_agent",
        capabilities=capabilities
    )

@pytest.fixture
def test_context():
    """Fixture for chain context"""
    return ChainContext(
        document_id="doc_test_123",
        task_type="test_task",
        start_time=datetime.utcnow().isoformat(),
        total_segments=1,
        standards_in_scope=["IFRS 9"],
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
def test_communication_unit():
    """Fixture for communication unit"""
    return CommunicationUnit(
        id="cu_test_456",
        type=CommunicationType.EVIDENCE,
        source_agent="previous_agent",
        timestamp=datetime.utcnow().isoformat(),
        segment_id="seg_789",
        segment_type="financial_statement",
        segment_content="Test financial data",
        evidence=[],
        previous_findings=[],
        accumulated_context={}
    )

@pytest.mark.unit
class TestBaseAgent:
    """Test cases for BaseAgent class"""
    
    def test_agent_initialization(self, test_agent):
        """Test agent initialization"""
        assert test_agent.name == "test_agent"
        assert len(test_agent.capabilities) == 1
        assert test_agent.capabilities[0].name == "test_capability"
        assert test_agent.next_agent is None
        assert test_agent.previous_agent is None
    
    def test_agent_capabilities(self, test_agent):
        """Test agent capabilities"""
        capability = test_agent.capabilities[0]
        assert capability.required_inputs == ["test_input"]
        assert capability.produced_outputs == ["test_output"]
        assert capability.description == "Test capability"
    
    @pytest.mark.asyncio
    async def test_process_without_input(self, test_agent, test_context):
        """Test processing without input communication unit"""
        response = await test_agent.process(None, test_context)
        assert response.success
        assert response.communication_unit is not None
        assert response.communication_unit.source_agent == "test_agent"
        assert len(response.communication_unit.evidence) == 1
    
    @pytest.mark.asyncio
    async def test_process_with_input(
        self,
        test_agent,
        test_context,
        test_communication_unit
    ):
        """Test processing with input communication unit"""
        response = await test_agent.process(test_communication_unit, test_context)
        assert response.success
        assert response.communication_unit is not None
        assert response.communication_unit.source_agent == "test_agent"
        assert len(response.communication_unit.evidence) == 1
    
    def test_validate_input_none(self, test_agent):
        """Test input validation with None"""
        assert test_agent._validate_input(None) is True
    
    def test_validate_input_valid(self, test_agent, test_communication_unit):
        """Test input validation with valid communication unit"""
        assert test_agent._validate_input(test_communication_unit) is True
    
    def test_validate_input_invalid(self, test_agent):
        """Test input validation with invalid communication unit"""
        with pytest.raises(ValidationError):
            CommunicationUnit(
                id="",  # Invalid: empty ID
                type=CommunicationType.EVIDENCE,
                source_agent="",  # Invalid: empty source
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_123",
                segment_type="test_type",
                segment_content="test_content",
                evidence=[]
            )
    
    def test_validate_output_valid(self, test_agent):
        """Test output validation with valid communication unit"""
        valid_cu = CommunicationUnit(
            id="test_cu",
            type=CommunicationType.EVIDENCE,
            source_agent="test_agent",
            timestamp=datetime.utcnow().isoformat(),
            segment_id="seg_123",
            segment_type="test_type",
            segment_content="test_content",
            evidence=[
                Evidence(
                    source_location="test",
                    content="test",
                    confidence=0.9
                )
            ]
        )
        assert test_agent._validate_output(valid_cu) is True
    
    def test_validate_output_invalid(self, test_agent):
        """Test output validation with invalid communication unit"""
        invalid_cu = CommunicationUnit(
            id="test_cu",
            type=CommunicationType.EVIDENCE,
            source_agent="test_agent",
            timestamp=datetime.utcnow().isoformat(),
            segment_id="seg_123",
            segment_type="test_type",
            segment_content="test_content",
            evidence=[]  # Invalid: empty evidence for EVIDENCE type
        )
        assert test_agent._validate_output(invalid_cu) is False
    
    def test_update_context(self, test_agent):
        """Test context updating"""
        current_context = {"existing": "data"}
        processing_result = AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="test_cu",
                type=CommunicationType.EVIDENCE,
                source_agent="test_agent",
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_123",
                segment_type="test_type",
                segment_content="test_content",
                evidence=[
                    Evidence(
                        source_location="test",
                        content="new evidence",
                        confidence=0.9
                    )
                ]
            )
        )
        
        updated_context = test_agent._update_context(
            current_context,
            processing_result
        )
        
        assert "existing" in updated_context
        assert "latest_evidence" in updated_context
        assert len(updated_context["latest_evidence"]) == 1
    
    @pytest.mark.asyncio
    async def test_chain_connection(self, test_agent, test_context):
        """Test agent chain connection"""
        next_agent = TestAgent(
            name="next_agent",
            capabilities=[]
        )
        test_agent.next_agent = next_agent
        next_agent.previous_agent = test_agent
        
        assert test_agent.next_agent.name == "next_agent"
        assert next_agent.previous_agent.name == "test_agent"
        
        # Test processing flows to next agent
        response = await test_agent.process(None, test_context)
        assert response.communication_unit.target_agent == "next_agent"
