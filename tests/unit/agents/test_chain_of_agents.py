"""Unit tests for ChainOfAgents class"""
import pytest
from datetime import datetime
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.chain_of_agents import ChainOfAgents
from src.agents.worker_agent import WorkerAgent
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
def mock_document():
    """Fixture for test document"""
    return {
        "id": "doc_test_001",
        "content": """
        Financial Statement Section 1
        Revenue: $5,000,000
        Operating Costs: $3,500,000
        
        Financial Statement Section 2
        Lease Agreements: Details of property leases...
        Contract Assets: Classification of assets...
        """,
        "type": "financial_statement"
    }

@pytest.fixture
def mock_standards():
    """Fixture for IFRS standards"""
    return ["IFRS 15", "IFRS 16"]

@pytest.fixture
def mock_fiscal_period():
    """Fixture for fiscal period"""
    return {
        "start": "2024-01-01",
        "end": "2024-12-31"
    }

@pytest.fixture
def mock_company_info():
    """Fixture for company information"""
    return {
        "name": "Test Corporation",
        "industry": "Technology",
        "size": "Large"
    }

@pytest.fixture
def chain_of_agents():
    """Fixture for chain of agents"""
    return ChainOfAgents(
        num_workers=2,
        segment_size=1000,
        config={
            "llm_model": "gpt-4",
            "temperature": 0.3
        }
    )

@pytest.mark.unit
class TestChainOfAgents:
    """Test cases for ChainOfAgents class"""
    
    def test_chain_initialization(self, chain_of_agents):
        """Test chain initialization"""
        assert len(chain_of_agents.workers) == 2
        assert isinstance(chain_of_agents.manager, ManagerAgent)
        assert chain_of_agents.segment_size == 1000
        assert "llm_model" in chain_of_agents.config
    
    def test_worker_chain_connection(self, chain_of_agents):
        """Test worker chain connections"""
        # Check worker connections
        for i in range(len(chain_of_agents.workers) - 1):
            assert chain_of_agents.workers[i].next_agent == chain_of_agents.workers[i + 1]
        
        # Check final worker connects to manager
        assert chain_of_agents.workers[-1].next_agent == chain_of_agents.manager
    
    @pytest.mark.asyncio
    async def test_segment_document(self, chain_of_agents, mock_document):
        """Test document segmentation"""
        segments = await chain_of_agents._segment_document(mock_document)
        
        assert len(segments) > 0
        assert all(isinstance(seg, dict) for seg in segments)
        assert all("id" in seg for seg in segments)
        assert all("content" in seg for seg in segments)
        assert all(len(seg["content"]) <= chain_of_agents.segment_size 
                  for seg in segments)
    
    def test_estimate_segments(self, chain_of_agents, mock_document):
        """Test segment estimation"""
        num_segments = chain_of_agents._estimate_segments(mock_document)
        
        assert num_segments > 0
        assert isinstance(num_segments, int)
        expected_segments = (len(mock_document["content"]) + 
                           chain_of_agents.segment_size - 1) // chain_of_agents.segment_size
        assert num_segments == expected_segments
    
    @pytest.mark.asyncio
    async def test_create_initial_cu(self, chain_of_agents):
        """Test initial communication unit creation"""
        segment = {
            "id": "seg_0",
            "content": "Test content",
            "start_pos": 0,
            "end_pos": 11
        }
        
        cu = await chain_of_agents._create_initial_cu(segment)
        
        assert cu.id is not None
        assert cu.type == CommunicationType.EVIDENCE
        assert cu.source_agent == "chain_orchestrator"
        assert cu.segment_content == "Test content"
        assert not cu.evidence
        assert not cu.previous_findings
    
    def test_create_segment_cu(self, chain_of_agents):
        """Test segment communication unit creation"""
        segment = {
            "id": "seg_1",
            "content": "New segment",
            "start_pos": 12,
            "end_pos": 23
        }
        previous_cu = CommunicationUnit(
            id="prev_cu",
            type=CommunicationType.EVIDENCE,
            source_agent="worker_1",
            timestamp=datetime.utcnow().isoformat(),
            segment_id="seg_123",
            segment_type="financial_statement",
            segment_content="Previous segment content",
            evidence=[
                Evidence(
                    source_location="test",
                    content="test evidence",
                    confidence=0.9
                )
            ],
            previous_findings=[{
                "agent": "worker_0",
                "evidence": [
                    Evidence(
                        source_location="test",
                        content="previous evidence",
                        confidence=0.9
                    )
                ]
            }],
            accumulated_context={"key": "value"}
        )
        
        cu = chain_of_agents._create_segment_cu(segment, previous_cu)
        
        assert cu.id is not None
        assert cu.segment_id == "seg_1"
        assert cu.segment_content == "New segment"
        assert cu.evidence == previous_cu.evidence
        assert cu.previous_findings == previous_cu.previous_findings
        assert cu.accumulated_context == previous_cu.accumulated_context
    
    @pytest.mark.asyncio
    async def test_process_document(
        self,
        chain_of_agents,
        mock_document,
        mock_standards,
        mock_fiscal_period,
        mock_company_info
    ):
        """Test complete document processing"""
        # Mock worker response
        mock_worker_response = AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="cu_worker_123",
                type=CommunicationType.EVIDENCE,
                source_agent="worker_0",
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_0",
                segment_type="document_segment",
                segment_content="Test content",
                evidence=[
                    Evidence(
                        source_location="test",
                        content="test evidence",
                        confidence=0.9
                    )
                ],
                accumulated_context={"findings": ["Test finding"]}
            )
        )
        chain_of_agents.workers[0].process = AsyncMock(return_value=mock_worker_response)
        
        # Mock manager response
        mock_manager_response = AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="cu_manager_123",
                type=CommunicationType.SYNTHESIS,
                source_agent="synthesis_manager",
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_999",
                segment_type="synthesis",
                segment_content="Final synthesis content",
                evidence=[
                    Evidence(
                        source_location="test",
                        content="test evidence",
                        confidence=0.9
                    )
                ],
                analysis=Analysis(
                    metrics={"test_metric": 0.9},
                    findings=["Test finding"],
                    implications=["Test implication"]
                )
            )
        )
        chain_of_agents.manager.process = AsyncMock(return_value=mock_manager_response)
        
        results = await chain_of_agents.process_document(
            document=mock_document,
            task_type="ifrs_compliance",
            standards=mock_standards,
            fiscal_period=mock_fiscal_period,
            company_info=mock_company_info
        )
        
        assert results["success"]
        assert "results" in results
        assert "metrics" in results["results"]
        assert "findings" in results["results"]
        assert "implications" in results["results"]
        assert "evidence" in results["results"]
    
    @pytest.mark.asyncio
    async def test_error_handling(
        self,
        chain_of_agents,
        mock_document,
        mock_standards,
        mock_fiscal_period,
        mock_company_info
    ):
        """Test error handling in document processing"""
        # Simulate error in worker processing
        chain_of_agents.workers[0].process = AsyncMock(
            side_effect=Exception("Worker processing error")
        )
        
        results = await chain_of_agents.process_document(
            document=mock_document,
            task_type="ifrs_compliance",
            standards=mock_standards,
            fiscal_period=mock_fiscal_period,
            company_info=mock_company_info
        )
        
        assert not results["success"]
        assert "error" in results
        assert "Worker processing error" in results["error"]
    
    @pytest.mark.asyncio
    async def test_empty_document(self, chain_of_agents, mock_fiscal_period, mock_company_info):
        """Test processing empty document"""
        empty_doc = {
            "id": "doc_empty_001",
            "content": "",
            "type": "financial_statement"
        }
        
        # Mock worker response for empty document
        mock_worker_response = AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="cu_worker_123",
                type=CommunicationType.EVIDENCE,
                source_agent="worker_0",
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_0",
                segment_type="document_segment",
                segment_content="Empty document",
                evidence=[
                    Evidence(
                        source_location="empty_doc",
                        content="Document has no content",
                        confidence=1.0
                    )
                ],
                accumulated_context={"findings": ["Document is empty"]}
            )
        )
        chain_of_agents.workers[0].process = AsyncMock(return_value=mock_worker_response)
        
        # Mock manager response for empty document
        mock_manager_response = AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="cu_manager_123",
                type=CommunicationType.SYNTHESIS,
                source_agent="synthesis_manager",
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_999",
                segment_type="synthesis",
                segment_content="Empty document synthesis",
                evidence=[
                    Evidence(
                        source_location="empty_doc",
                        content="Document has no content",
                        confidence=1.0
                    )
                ],
                analysis=Analysis(
                    metrics={"content_length": 0},
                    findings=["Document is empty"],
                    implications=["Cannot perform IFRS compliance analysis"]
                )
            )
        )
        chain_of_agents.manager.process = AsyncMock(return_value=mock_manager_response)
        
        results = await chain_of_agents.process_document(
            document=empty_doc,
            task_type="ifrs_compliance",
            standards=["IFRS 15"],
            fiscal_period=mock_fiscal_period,
            company_info=mock_company_info
        )
        
        # Debug output
        print("\nEmpty document test results:")
        print(f"Success: {results.get('success')}")
        print(f"Error: {results.get('error')}")
        if 'results' in results:
            print("Results structure:")
            print(f"  metrics: {results['results'].get('metrics')}")
            print(f"  findings: {results['results'].get('findings')}")
            print(f"  implications: {results['results'].get('implications')}")
            print(f"  evidence: {results['results'].get('evidence')}")
        
        # Check basic structure
        assert results["success"]  # Should handle empty document gracefully
        assert "results" in results
        
        # Check results structure
        assert isinstance(results["results"]["metrics"], dict)
        assert isinstance(results["results"]["findings"], list)
        assert isinstance(results["results"]["implications"], list)
        assert isinstance(results["results"]["evidence"], list)
        
        # Check specific values for empty document
        assert results["results"]["metrics"] == {"content_length": 0}
        assert results["results"]["findings"] == ["Document is empty"]
        assert results["results"]["implications"] == ["Cannot perform IFRS compliance analysis"]
        assert len(results["results"]["evidence"]) == 1
        assert results["results"]["evidence"][0]["content"] == "Document has no content"
        assert results["results"]["evidence"][0]["confidence"] == 1.0
    
    @pytest.mark.asyncio
    async def test_result_extraction(
        self,
        chain_of_agents,
        mock_document,
        mock_standards,
        mock_fiscal_period,
        mock_company_info
    ):
        """Test extraction of final results"""
        # Mock worker response
        mock_worker_response = AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="cu_worker_123",
                type=CommunicationType.EVIDENCE,
                source_agent="worker_0",
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_0",
                segment_type="document_segment",
                segment_content="Test content",
                evidence=[
                    Evidence(
                        source_location="test",
                        content="test evidence",
                        confidence=0.9
                    )
                ],
                accumulated_context={"findings": ["Test finding"]}
            )
        )
        chain_of_agents.workers[0].process = AsyncMock(return_value=mock_worker_response)
        
        # Mock manager response
        mock_manager_response = AgentResponse(
            success=True,
            communication_unit=CommunicationUnit(
                id="cu_manager_123",
                type=CommunicationType.SYNTHESIS,
                source_agent="synthesis_manager",
                timestamp=datetime.utcnow().isoformat(),
                segment_id="seg_999",
                segment_type="synthesis",
                segment_content="Final synthesis content",
                evidence=[
                    Evidence(
                        source_location="test",
                        content="test evidence",
                        confidence=0.9
                    )
                ],
                analysis=Analysis(
                    metrics={"test_metric": 0.9},
                    findings=["Test finding"],
                    implications=["Test implication"]
                )
            )
        )
        chain_of_agents.manager.process = AsyncMock(return_value=mock_manager_response)
        
        # Process document
        results = await chain_of_agents.process_document(
            document=mock_document,
            task_type="ifrs_compliance",
            standards=mock_standards,
            fiscal_period=mock_fiscal_period,
            company_info=mock_company_info
        )
        
        # Check results structure
        assert results["success"]
        assert isinstance(results["results"]["metrics"], dict)
        assert isinstance(results["results"]["findings"], list)
        assert isinstance(results["results"]["implications"], list)
        assert isinstance(results["results"]["evidence"], list)
        
        # Check evidence format
        for evidence in results["results"]["evidence"]:
            assert "content" in evidence
            assert "location" in evidence
            assert "confidence" in evidence
            assert 0 <= evidence["confidence"] <= 1
            
        # Check specific values
        assert results["results"]["metrics"] == {"test_metric": 0.9}
        assert results["results"]["findings"] == ["Test finding"]
        assert results["results"]["implications"] == ["Test implication"]
