"""
Langflow Integration for Chain of Agents
"""
from typing import List, Dict, Any, Optional
import json
from langflow import CustomComponent
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor

from .langchain_orchestrator import (
    LangchainOrchestrator,
    IFRSAnalysisTool,
    IFRSSynthesisTool
)

class IFRSAnalysisComponent(CustomComponent):
    """Langflow component for IFRS document analysis"""
    
    display_name = "IFRS Analysis"
    description = "Analyzes IFRS documents for compliance"
    
    def build_config(self):
        """Build component configuration"""
        return {
            "document": {
                "type": "file",
                "required": True,
                "description": "IFRS document to analyze",
                "file_types": [".txt", ".pdf", ".doc", ".docx"],
                "max_size_mb": 10
            },
            "standards": {
                "type": "list",
                "required": True,
                "description": "IFRS standards to check",
                "items": {
                    "type": "str",
                    "pattern": "^IFRS\s\d+$"
                }
            },
            "fiscal_period_start": {
                "type": "str",
                "required": True,
                "description": "Start of fiscal period (YYYY-MM-DD)",
                "pattern": "^\d{4}-\d{2}-\d{2}$"
            },
            "fiscal_period_end": {
                "type": "str",
                "required": True,
                "description": "End of fiscal period (YYYY-MM-DD)",
                "pattern": "^\d{4}-\d{2}-\d{2}$"
            },
            "company_name": {
                "type": "str",
                "required": True,
                "description": "Company name",
                "min_length": 1,
                "max_length": 100
            },
            "company_industry": {
                "type": "str",
                "required": True,
                "description": "Company industry",
                "min_length": 1,
                "max_length": 50
            },
            "num_workers": {
                "type": "int",
                "required": False,
                "default": 3,
                "description": "Number of worker agents",
                "min": 1,
                "max": 10
            },
            "segment_size": {
                "type": "int",
                "required": False,
                "default": 1000,
                "description": "Size of document segments",
                "min": 100,
                "max": 5000
            }
        }
    
    def build(
        self,
        document: str,
        standards: List[str],
        fiscal_period_start: str,
        fiscal_period_end: str,
        company_name: str,
        company_industry: str,
        num_workers: int = 3,
        segment_size: int = 1000
    ) -> Dict[str, Any]:
        """Build the analysis pipeline"""
        try:
            # Validate inputs
            import re
            from datetime import datetime
            
            # Validate document
            if not os.path.exists(document):
                raise ValueError(f"Document does not exist: {document}")
            file_size = os.path.getsize(document) / (1024 * 1024)  # MB
            if file_size > 10:
                raise ValueError(f"Document size ({file_size:.1f}MB) exceeds 10MB limit")
            
            # Validate standards
            for std in standards:
                if not re.match(r'^IFRS\s\d+$', std):
                    raise ValueError(f"Invalid IFRS standard format: {std}. Must be 'IFRS X' where X is a number")
            
            # Validate dates
            try:
                start_date = datetime.strptime(fiscal_period_start, '%Y-%m-%d')
                end_date = datetime.strptime(fiscal_period_end, '%Y-%m-%d')
                if end_date <= start_date:
                    raise ValueError("Fiscal period end must be after start")
            except ValueError as e:
                raise ValueError(f"Invalid date format: {str(e)}")
            
            # Validate other parameters
            if not (1 <= num_workers <= 10):
                raise ValueError(f"Number of workers must be between 1 and 10, got {num_workers}")
            if not (100 <= segment_size <= 5000):
                raise ValueError(f"Segment size must be between 100 and 5000, got {segment_size}")
            if not (1 <= len(company_name) <= 100):
                raise ValueError(f"Company name length must be between 1 and 100 characters")
            if not (1 <= len(company_industry) <= 50):
                raise ValueError(f"Company industry length must be between 1 and 50 characters")
            
            # Create orchestrator
            orchestrator = LangchainOrchestrator(
                num_workers=num_workers,
                segment_size=segment_size
            )
            
            # Read document
            try:
                with open(document, 'r') as f:
                    content = f.read()
            except Exception as e:
                raise IOError(f"Failed to read document: {str(e)}")
            
            # Process document
            results = orchestrator.process_document(
                document={
                    "id": "doc_001",
                    "content": content
                },
                task_type="ifrs_compliance",
                standards=standards,
                fiscal_period={
                    "start": fiscal_period_start,
                    "end": fiscal_period_end
                },
                company_info={
                    "name": company_name,
                    "industry": company_industry
                }
            )
            
            if not results.get("success"):
                raise Exception(results.get("error", "Unknown error during document processing"))
            
            return {
                "success": True,
                "results": results["results"],
                "metadata": {
                    "document_size": file_size,
                    "num_workers": num_workers,
                    "segment_size": segment_size,
                    "standards_analyzed": standards
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

class IFRSToolComponent(CustomComponent):
    """Langflow component for IFRS analysis tools"""
    
    display_name = "IFRS Tools"
    description = "Tools for IFRS document analysis"
    
    def build_config(self):
        """Build component configuration"""
        return {
            "tool_type": {
                "type": "str",
                "required": True,
                "options": ["analysis", "synthesis"],
                "description": "Type of IFRS tool",
                "default": "analysis"
            },
            "worker_config": {
                "type": "dict",
                "required": False,
                "description": "Configuration for worker agent",
                "properties": {
                    "model": {
                        "type": "str",
                        "description": "LLM model to use",
                        "default": "gpt-4"
                    },
                    "temperature": {
                        "type": "float",
                        "description": "LLM temperature",
                        "min": 0.0,
                        "max": 1.0,
                        "default": 0.3
                    },
                    "max_tokens": {
                        "type": "int",
                        "description": "Maximum tokens per response",
                        "min": 100,
                        "max": 4000,
                        "default": 2000
                    }
                }
            }
        }
    
    def build(
        self,
        tool_type: str,
        worker_config: Optional[Dict] = None
    ) -> BaseTool:
        """Build the IFRS tool"""
        try:
            # Validate tool type
            if tool_type not in ["analysis", "synthesis"]:
                raise ValueError(f"Invalid tool type: {tool_type}. Must be 'analysis' or 'synthesis'")
            
            # Validate worker config if provided
            if worker_config:
                if not isinstance(worker_config, dict):
                    raise TypeError(f"worker_config must be a dictionary, got {type(worker_config)}")
                
                # Validate model
                model = worker_config.get("model", "gpt-4")
                if not isinstance(model, str):
                    raise TypeError(f"model must be a string, got {type(model)}")
                
                # Validate temperature
                temp = worker_config.get("temperature", 0.3)
                if not isinstance(temp, (int, float)) or not 0 <= temp <= 1:
                    raise ValueError(f"temperature must be between 0 and 1, got {temp}")
                
                # Validate max_tokens
                max_tokens = worker_config.get("max_tokens", 2000)
                if not isinstance(max_tokens, int) or not 100 <= max_tokens <= 4000:
                    raise ValueError(f"max_tokens must be between 100 and 4000, got {max_tokens}")
            
            # Create appropriate tool based on type
            if tool_type == "analysis":
                from src.agents.worker_agent import WorkerAgent
                try:
                    worker = WorkerAgent(
                        name="ifrs_worker",
                        capabilities=["document_analysis", "ifrs_compliance"],
                        config=worker_config
                    )
                    return IFRSAnalysisTool(worker)
                except Exception as e:
                    raise Exception(f"Failed to create analysis tool: {str(e)}")
            else:
                from src.agents.manager_agent import ManagerAgent
                try:
                    manager = ManagerAgent(
                        name="ifrs_manager",
                        capabilities=["synthesis", "recommendation"],
                        config=worker_config
                    )
                    return IFRSSynthesisTool(manager)
                except Exception as e:
                    raise Exception(f"Failed to create synthesis tool: {str(e)}")
                    
        except Exception as e:
            raise Exception(f"Failed to build IFRS tool: {str(e)}")

class IFRSAgentComponent(CustomComponent):
    """Langflow component for IFRS analysis agent"""
    
    display_name = "IFRS Agent"
    description = "Agent for orchestrating IFRS document analysis"
    
    def build_config(self):
        """Build component configuration"""
        return {
            "tools": {
                "type": "list",
                "required": True,
                "description": "List of IFRS tools",
                "min_items": 1,
                "items": {
                    "type": "tool",
                    "tool_types": ["IFRSAnalysisTool", "IFRSSynthesisTool"]
                }
            },
            "memory": {
                "type": "memory",
                "required": False,
                "description": "Memory for the agent",
                "memory_type": "ConversationBufferMemory"
            },
            "agent_config": {
                "type": "dict",
                "required": False,
                "description": "Configuration for the agent",
                "properties": {
                    "model": {
                        "type": "str",
                        "description": "LLM model to use",
                        "default": "gpt-4"
                    },
                    "temperature": {
                        "type": "float",
                        "description": "LLM temperature",
                        "min": 0.0,
                        "max": 1.0,
                        "default": 0.3
                    },
                    "max_tokens": {
                        "type": "int",
                        "description": "Maximum tokens per response",
                        "min": 100,
                        "max": 4000,
                        "default": 2000
                    },
                    "verbose": {
                        "type": "bool",
                        "description": "Enable verbose output",
                        "default": True
                    }
                }
            }
        }
    
    def build(
        self,
        tools: List[BaseTool],
        memory: Optional[Any] = None,
        agent_config: Optional[Dict] = None
    ) -> AgentExecutor:
        """Build the IFRS agent"""
        try:
            # Validate tools
            if not tools:
                raise ValueError("At least one tool must be provided")
            
            valid_tool_types = {"IFRSAnalysisTool", "IFRSSynthesisTool"}
            for tool in tools:
                if not isinstance(tool, BaseTool):
                    raise TypeError(f"Invalid tool type: {type(tool)}. Must be a BaseTool instance")
                if tool.__class__.__name__ not in valid_tool_types:
                    raise ValueError(f"Invalid tool: {tool.__class__.__name__}. Must be one of {valid_tool_types}")
            
            # Validate memory if provided
            if memory:
                from langchain.memory import ConversationBufferMemory
                if not isinstance(memory, ConversationBufferMemory):
                    raise TypeError(f"Invalid memory type: {type(memory)}. Must be ConversationBufferMemory")
            
            # Validate agent config if provided
            if agent_config:
                if not isinstance(agent_config, dict):
                    raise TypeError(f"agent_config must be a dictionary, got {type(agent_config)}")
                
                # Validate model
                model = agent_config.get("model", "gpt-4")
                if not isinstance(model, str):
                    raise TypeError(f"model must be a string, got {type(model)}")
                
                # Validate temperature
                temp = agent_config.get("temperature", 0.3)
                if not isinstance(temp, (int, float)) or not 0 <= temp <= 1:
                    raise ValueError(f"temperature must be between 0 and 1, got {temp}")
                
                # Validate max_tokens
                max_tokens = agent_config.get("max_tokens", 2000)
                if not isinstance(max_tokens, int) or not 100 <= max_tokens <= 4000:
                    raise ValueError(f"max_tokens must be between 100 and 4000, got {max_tokens}")
                
                # Validate verbose
                verbose = agent_config.get("verbose", True)
                if not isinstance(verbose, bool):
                    raise TypeError(f"verbose must be a boolean, got {type(verbose)}")
            
            # Create orchestrator with validated config
            try:
                orchestrator = LangchainOrchestrator(
                    config=agent_config
                )
                agent_chain = orchestrator._create_agent_chain()
                
                # Add tools and memory to agent chain
                agent_chain.tools = tools
                if memory:
                    agent_chain.memory = memory
                
                return agent_chain
                
            except Exception as e:
                raise Exception(f"Failed to create agent chain: {str(e)}")
            
        except Exception as e:
            raise Exception(f"Failed to build IFRS agent: {str(e)}")

def register_components():
    """Register Langflow components"""
    return {
        "IFRSAnalysis": IFRSAnalysisComponent,
        "IFRSTools": IFRSToolComponent,
        "IFRSAgent": IFRSAgentComponent
    }
