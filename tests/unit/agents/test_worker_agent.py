"""Unit tests for WorkerAgent class"""
import pytest
from datetime import datetime
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.worker_agent import WorkerAgent
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
    - revenue: 1000000
    - profit_margin: 15.5
    
    FINDINGS:
    - Revenue recognition complies with IFRS 15
    - Lease accounting needs review per IFRS 16
    
    EVIDENCE:
    - Revenue properly recognized over time [section 3.1] [0.95]
    - Lease terms not properly disclosed [section 4.2] [0.85]
    
    IMPLICATIONS:
    - Current revenue recognition practices are adequate
    - Need to enhance lease disclosure documentation
    """

@pytest.fixture
def worker_agent():
    """Fixture for worker agent"""
    agent = WorkerAgent(
        name="test_worker",
        capabilities=["document_analysis", "ifrs_compliance"],
        segment_size=1000
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
        total_segments=1,
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
def test_communication_unit():
    """Fixture for input communication unit"""
    return CommunicationUnit(
        id="test_cu",
        type=CommunicationType.EVIDENCE,
        source_agent="previous_agent",
        timestamp=datetime.utcnow().isoformat(),
        segment_id="test_segment",
        segment_type="financial_statement",
        segment_content="Test financial data showing revenue of 1M",
        evidence=[],
        previous_findings=[
            {"content": "Previous finding about cash flow"}
        ],
        accumulated_context={}
    )

@pytest.mark.unit
class TestWorkerAgent:
    """Test cases for WorkerAgent class"""
    
    def test_worker_initialization(self, worker_agent):
        """Test worker agent initialization"""
        assert worker_agent.name == "test_worker"
        assert "document_analysis" in worker_agent.capabilities
        assert "ifrs_compliance" in worker_agent.capabilities
        assert worker_agent.segment_size == 1000
    
    @pytest.mark.asyncio
    async def test_process_segment(
        self,
        worker_agent,
        test_context,
        test_communication_unit,
        mock_llm_response
    ):
        """Test segment processing"""
        response = await worker_agent._process_segment(
            test_communication_unit,
            test_context
        )
        
        assert response.success
        assert response.communication_unit is not None
        assert response.communication_unit.type == CommunicationType.ANALYSIS
        assert len(response.communication_unit.evidence) == 2
        assert "revenue" in response.communication_unit.analysis.metrics
    
    def test_parse_llm_response(self, worker_agent, mock_llm_response):
        """Test LLM response parsing"""
        findings, evidence = worker_agent._parse_llm_response(mock_llm_response)
        
        assert len(findings) == 2
        assert "Revenue recognition complies with IFRS 15" in findings
        assert len(evidence) == 2
        assert evidence[0].source_location == "section 3.1"
        assert evidence[0].confidence == 0.95
    
    def test_extract_metrics(self, worker_agent):
        """Test metrics extraction from findings"""
        findings = [
            "revenue: 1000000",
            "profit_margin: 15.5",
            "non_metric finding"
        ]
        
        metrics = worker_agent._extract_metrics(findings)
        
        assert len(metrics) == 2
        assert metrics["revenue"] == 1000000
        assert metrics["profit_margin"] == 15.5
    
    def test_derive_implications(self, worker_agent):
        """Test implications derivation"""
        findings = [
            "Revenue recognition complies with IFRS 15",
            "Lease accounting needs review"
        ]
        
        implications = worker_agent._derive_implications(findings)
        
        assert len(implications) == 2
        assert all(isinstance(imp, str) for imp in implications)
    
    def test_format_previous_findings(self, worker_agent):
        """Test previous findings formatting"""
        findings = [
            {"content": "Finding 1"},
            {"content": "Finding 2"}
        ]
        
        formatted = worker_agent._format_previous_findings(findings)
        
        assert "1. Finding 1" in formatted
        assert "2. Finding 2" in formatted
    
    def test_update_context(self, worker_agent):
        """Test context updating with new findings"""
        current_context = {
            "findings": ["Old finding"]
        }
        new_findings = ["New finding 1", "New finding 2"]
        
        updated_context = worker_agent._update_context(
            current_context,
            new_findings
        )
        
        assert len(updated_context["findings"]) == 3
        assert "New finding 1" in updated_context["findings"]
        assert "Old finding" in updated_context["findings"]
    
    @pytest.mark.asyncio
    async def test_generate_segment_prompt(
        self,
        worker_agent,
        test_context,
        test_communication_unit
    ):
        """Test prompt generation"""
        prompt = worker_agent._generate_segment_prompt(
            test_communication_unit.segment_content,
            test_communication_unit.previous_findings,
            test_context
        )
        
        assert "Test financial data" in prompt
        assert "IFRS 15" in prompt
        assert "IFRS 16" in prompt
        assert "Previous finding about cash flow" in prompt
    
    @pytest.mark.asyncio
    async def test_process_with_error_handling(
        self,
        worker_agent,
        test_context,
        test_communication_unit
    ):
        """Test error handling during processing"""
        # Mock LLM to raise an exception
        worker_agent.llm.get_completion = AsyncMock(
            side_effect=Exception("LLM error")
        )
        
        response = await worker_agent.process(
            test_communication_unit,
            test_context
        )
        
        assert not response.success
        assert "LLM error" in response.error
    
    @pytest.mark.asyncio
    async def test_process_empty_segment(
        self,
        worker_agent,
        test_context
    ):
        """Test processing with empty segment"""
        empty_cu = CommunicationUnit(
            id="empty_cu",
            type=CommunicationType.EVIDENCE,
            source_agent="previous_agent",
            segment_content="",
            evidence=[]
        )
        
        response = await worker_agent.process(empty_cu, test_context)
        
        assert response.success  # Should handle empty segments gracefully
        assert response.communication_unit is not None
