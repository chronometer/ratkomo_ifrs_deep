"""
Langchain Orchestrator for Chain of Agents
"""
from typing import List, Dict, Any, Optional
import json
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

from src.agents.worker_agent import WorkerAgent
from src.agents.manager_agent import ManagerAgent
from src.agents.communication import ChainContext, CommunicationUnit, CommunicationType
from src.config.llm_config import get_llm_config

from pydantic import BaseModel, Field

class IFRSAnalysisTool(BaseTool, BaseModel):
    """Tool for IFRS document analysis"""
    name: str = Field(default="ifrs_analysis", description="Tool name")
    description: str = Field(
        default="Analyzes document segments for IFRS compliance. Input should be a JSON string containing segment_content, standards, and fiscal_period.",
        description="Tool description"
    )
    worker: WorkerAgent = Field(description="Worker agent for analysis")
    
    def __init__(self, worker: WorkerAgent):
        super().__init__(worker=worker)
        
    async def _arun(
        self,
        tool_input: str
    ) -> Dict[str, Any]:
        """Run the tool asynchronously"""
        try:
            # Parse input JSON
            import json
            input_data = json.loads(tool_input)
            
            # Create context
            context = ChainContext(
                document_id="doc_001",
                task_type="ifrs_compliance",
                start_time=self.worker._get_timestamp(),
                total_segments=1,
                standards_in_scope=input_data.get('standards', []),
                fiscal_period=input_data.get('fiscal_period', {'start': '2023-01-01', 'end': '2023-12-31'}),
                company_info=input_data.get('company_info', {'name': 'Pihlajalinna', 'industry': 'Healthcare'})
            )
            
            # Create communication unit
            cu = CommunicationUnit(
                id=f"cu_{self.worker._generate_cu_id()}",
                type=CommunicationType.ANALYSIS,
                source_agent=self.worker.name,
                timestamp=self.worker._get_timestamp(),
                segment_id="seg_001",
                segment_type="financial_statement",
                segment_content=input_data.get('segment_content', ''),
                evidence=[],
                previous_findings=[],
                accumulated_context={}
            )
            
            # Process through worker
            response = await self.worker.process(cu, context)
            
            if not response.success:
                raise Exception(response.error)
                
            return {
                "findings": response.communication_unit.analysis.findings,
                "evidence": [
                    {
                        "content": e.content,
                        "confidence": e.confidence,
                        "location": e.source_location
                    }
                    for e in response.communication_unit.evidence
                ]
            }
        except Exception as e:
            return {"error": str(e)}
        
    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use arun instead")

class IFRSSynthesisTool(BaseTool, BaseModel):
    """Tool for synthesizing IFRS analysis results"""
    name: str = Field(default="ifrs_synthesis", description="Tool name")
    description: str = Field(
        default="Synthesizes findings from IFRS analysis into final recommendations. Input should be a JSON string containing findings and standards.",
        description="Tool description"
    )
    manager: ManagerAgent = Field(description="Manager agent for synthesis")
    
    def __init__(self, manager: ManagerAgent):
        super().__init__(manager=manager)
        
    async def _arun(
        self,
        tool_input: str
    ) -> Dict[str, Any]:
        """Run the tool asynchronously"""
        try:
            # Parse input JSON
            import json
            input_data = json.loads(tool_input)
            
            # Create context
            context = ChainContext(
                start_time=self.manager._get_timestamp(),
                total_segments=1,
                standards=input_data.get('standards', []),
                fiscal_period={}
            )
            
            # Create communication unit with accumulated findings
            cu = CommunicationUnit(
                id=self.manager._generate_cu_id(),
                type="synthesis",
                source_agent=self.manager.name,
                timestamp=self.manager._get_timestamp(),
                segment_id="final",
                segment_type="synthesis",
                segment_content="",
                evidence=[],
                previous_findings=input_data.get('findings', []),
                accumulated_context={"findings": input_data.get('findings', [])}
            )
            
            # Process through manager
            response = await self.manager.process(cu, context)
            
            if not response.success:
                raise Exception(response.error)
                
            return {
                "recommendations": response.communication_unit.analysis.recommendations,
                "summary": response.communication_unit.analysis.summary
            }
        except Exception as e:
            return {"error": str(e)}
            
    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use arun instead")

        
    def _run(self, *args, **kwargs):
        raise NotImplementedError("Use arun instead")

class LangchainOrchestrator:
    """
    Orchestrates the Chain of Agents using Langchain
    """
    
    def __init__(
        self,
        num_workers: int = 3,
        segment_size: int = 1000,
        config: Optional[Dict] = None
    ):
        self.config = config or {}
        self.num_workers = 5  # Fixed number of specialized workers
        self.segment_size = segment_size
        self.analysis_iterations = 3  # Number of analysis iterations per segment
        
        # Initialize specialized worker agents
        self.workers = [
            WorkerAgent(
                name="revenue_recognition_expert",
                capabilities=["document_analysis", "ifrs_compliance", "revenue_recognition"],
                segment_size=segment_size,
                config={**(self.config), "specialization": "revenue"}
            ),
            WorkerAgent(
                name="asset_lease_expert",
                capabilities=["document_analysis", "ifrs_compliance", "lease_analysis"],
                segment_size=segment_size,
                config={**(self.config), "specialization": "assets_leases"}
            ),
            WorkerAgent(
                name="financial_statement_expert",
                capabilities=["document_analysis", "ifrs_compliance", "financial_statements"],
                segment_size=segment_size,
                config={**(self.config), "specialization": "statements"}
            ),
            WorkerAgent(
                name="disclosure_expert",
                capabilities=["document_analysis", "ifrs_compliance", "disclosures"],
                segment_size=segment_size,
                config={**(self.config), "specialization": "disclosures"}
            ),
            WorkerAgent(
                name="industry_expert",
                capabilities=["document_analysis", "ifrs_compliance", "industry_specific"],
                segment_size=segment_size,
                config={**(self.config), "specialization": "healthcare_industry"}
            )
        ]
        
        # Initialize intermediate synthesis agents
        self.technical_synthesizer = ManagerAgent(
            name="technical_compliance_synthesizer",
            capabilities=["synthesis", "technical_compliance"],
            config={**(self.config), "synthesis_type": "technical"}
        )
        
        self.business_synthesizer = ManagerAgent(
            name="business_impact_synthesizer",
            capabilities=["synthesis", "business_impact"],
            config={**(self.config), "synthesis_type": "business"}
        )
        
        # Final synthesis manager
        self.manager = ManagerAgent(
            name="final_synthesis_manager",
            capabilities=["synthesis", "recommendation", "report_generation"],
            config=self.config
        )
        
        # Create tools with specialized analysis and synthesis
        self.tools = [
            IFRSAnalysisTool(worker) for worker in self.workers
        ] + [
            IFRSSynthesisTool(self.technical_synthesizer),
            IFRSSynthesisTool(self.business_synthesizer),
            IFRSSynthesisTool(self.manager)
        ]
        
        # Initialize LLM with OpenRouter
        llm_config = get_llm_config()
        self.llm = ChatOpenAI(
            model=llm_config.default_model,
            temperature=0.3,
            openai_api_key=llm_config.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
            max_tokens=1000,
            default_headers={
                "HTTP-Referer": "https://github.com/jargothia/Agent_chain_doc_analysis",  # Website URL
                "X-Title": "IFRS Analysis Agent"  # Name of your application
            }
        )
        
        # Initialize progress callback
        from src.utils.progress_callback import ProgressCallback
        self.progress_callback = ProgressCallback()
        
        # Initialize separate memories for analysis and synthesis
        from langchain_core.memory import BaseMemory
        from langchain.memory import ConversationBufferMemory
        
        self.analysis_memory = ConversationBufferMemory(
            memory_key="chat_history",  # Key must match memory_prompts in agent
            return_messages=True,
            output_key="output",  # Explicitly set output key
            max_token_limit=2000  # Limit memory size
        )
        
        self.synthesis_memory = ConversationBufferMemory(
            memory_key="chat_history",  # Key must match memory_prompts in agent
            return_messages=True,
            output_key="output",  # Explicitly set output key
            max_token_limit=2000  # Limit memory size
        )
        
        # Create separate agent chains for analysis and synthesis
        self.analysis_chain = self._create_agent_chain(
            memory=self.analysis_memory,
            system_message="""You are an IFRS analysis expert. Your task is to analyze document segments for IFRS compliance.
            
            For each segment analysis:
            1. Identify key IFRS-relevant information and disclosures
            2. Evaluate compliance with specific IFRS standards
            3. Note any potential gaps or areas needing clarification
            4. Consider industry-specific implications
            
            Structure your analysis as follows:
            1. Key Points: Bullet points of main findings
            2. Compliance Status: Clear statement of compliance level
            3. Recommendations: Specific suggestions for improvement
            4. References: Relevant IFRS standards and paragraphs
            
            IMPORTANT: Format your output in a clear, structured manner that can be easily synthesized later.""",
            max_iterations=5
        )
        
        self.synthesis_chain = self._create_agent_chain(
            memory=self.synthesis_memory,
            system_message="""You are an IFRS synthesis expert. Your task is to combine and synthesize analysis results into a comprehensive markdown report.
            
            When synthesizing technical findings:
            1. Focus on compliance gaps and recommendations
            2. Prioritize findings by severity and impact
            3. Provide specific references to IFRS standards
            
            When synthesizing business impact:
            1. Evaluate operational implications
            2. Assess financial impact
            3. Identify risks and opportunities
            
            For the final report:
            1. Create an executive summary highlighting key findings
            2. Organize technical findings by standard
            3. Present business impacts with clear action items
            4. Use proper markdown formatting with headers, lists, and emphasis
            
            IMPORTANT: Always structure your output in markdown format with clear sections and subsections.""",
            max_iterations=3
        )
        
    def _create_agent_chain(self, memory: ConversationBufferMemory, system_message: str, max_iterations: int = 5) -> AgentExecutor:
        """Create a Langchain agent chain with specific memory and system message
        
        Args:
            memory: Memory instance to use for this chain
            system_message: System message for the agent
            max_iterations: Maximum number of iterations before stopping
            
        Returns:
            AgentExecutor instance
        """
        # Create system message
        system_msg = SystemMessage(content=system_message)
        
        # Create agent executor with specific memory and iteration limit
        agent = self._create_agent(system_message=system_msg)
        
        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            callbacks=[self.progress_callback] if self.progress_callback else None,
            verbose=True,
            max_iterations=max_iterations,
            handle_parsing_errors=True,
            early_stopping_method="force",  # Force stop after max_iterations
            return_intermediate_steps=True  # Get detailed steps for better synthesis
        )
        
        return agent_executor
    
    def _create_agent(self, system_message: SystemMessage) -> BaseSingleActionAgent:
        """Create the Langchain agent with specific system message
        
        Args:
            system_message: System message defining the agent's role and task
            
        Returns:
            Agent instance
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.agents import create_openai_functions_agent
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message.content + """
            IMPORTANT: When using tools, always format the input as valid JSON. For example:
            
            For ifrs_analysis tool:
            {
                "segment_content": "<document text>",
                "standards": ["IFRS 15", "IFRS 16", "IAS 1"],
                "fiscal_period": {
                    "start": "2023-01-01",
                    "end": "2023-12-31"
                },
                "company_info": {
                    "name": "Pihlajalinna",
                    "industry": "Healthcare",
                    "jurisdiction": "FI"
                }
            }
            
            For ifrs_synthesis tool:
            {
                "findings": [<list of findings>],
                "standards": ["IFRS 15", "IFRS 16", "IAS 1"],
                "fiscal_period": {
                    "start": "2023-01-01",
                    "end": "2023-12-31"
                }
            }
            
            Let's approach this systematically:"""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content="{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        from langchain.agents import create_openai_functions_agent
        from langchain.agents.agent import AgentExecutor
        
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        return agent
        
    async def process_document(
        self,
        document: Dict[str, Any],
        task_type: str,
        standards: List[str],
        fiscal_period: Dict[str, str],
        company_info: Dict[str, Any],
        start_time: str,
        total_segments: int,
        callbacks: List[BaseCallbackHandler] = None
    ) -> Dict[str, Any]:
        """
        Process document through Langchain orchestration with multiple iterations and specialized analysis
        
        Args:
            document: Document to analyze
            task_type: Type of analysis task
            standards: IFRS standards to check
            fiscal_period: Fiscal period details
            company_info: Company information
            
        Returns:
            Dict containing detailed analysis results with technical and business insights
        """
        try:
            # Create chain context
            context = ChainContext(
                document_id=document.get("id", "doc_001"),
                task_type=task_type,
                standards_in_scope=standards,
                fiscal_period=fiscal_period,
                company_info=company_info,
                start_time=start_time,
                total_segments=total_segments
            )
            
            # Segment document
            segments = self._segment_document(document)
            
            # Set initial phase to analysis
            if self.progress_callback:
                self.progress_callback.current_phase = "analysis"
                
            # Process each segment with analysis chain
            segment_results = []
            for segment in segments:
                iteration_results = []
                
                # Multiple iterations per segment
                for iteration in range(self.analysis_iterations):
                    worker_results = []
                    
                    # Each specialized worker analyzes the segment
                    for worker in self.workers:
                        # Create communication unit
                        cu = CommunicationUnit(
                            id=f"cu_{segment['id']}_{worker.name}_{iteration}",
                            type=CommunicationType.ANALYSIS,
                            source_agent="orchestrator",
                            target_agent=worker.name,
                            timestamp=self._get_timestamp(),
                            segment_id=segment['id'],
                            segment_type=worker.config.get("specialization", "general"),
                            segment_content=segment['content'],
                            metadata={"iteration": iteration}
                        )
                        
                        # Run specialized analysis with analysis chain
                        result = await self.analysis_chain.ainvoke({
                            "input": json.dumps({
                                "segment_content": segment['content'],
                                "standards": standards,
                                "fiscal_period": fiscal_period,
                                "specialization": worker.config.get("specialization"),
                                "iteration": iteration,
                                "segment_id": segment['id']
                            }),
                            "chat_history": []  # Start with empty chat history
                        })
                        result = result.get("output", "")
                        
                        # Update progress
                        if self.progress_callback:
                            self.progress_callback.step_count += 1
                            total = self.progress_callback.total_tasks["analysis"]
                            print(f"\r📊 Document Analysis Progress: {self.progress_callback.step_count}/{total} tasks", end="")
                            sys.stdout.flush()
                        
                        worker_results.append({
                            "worker": worker.name,
                            "analysis": result,
                            "segment_id": segment['id'],
                            "iteration": iteration
                        })
                        
                        # Clear analysis memory after each worker
                        self.analysis_memory.clear()
                    
                    iteration_results.append({
                        "iteration": iteration,
                        "findings": worker_results
                    })
                
                segment_results.append({
                    "segment_id": segment['id'],
                    "content": segment['content'],
                    "iterations": iteration_results
                })
            
            # Switch to synthesis chain for higher-level analysis
            if self.progress_callback:
                self.progress_callback.current_phase = "synthesis"
                self.progress_callback.step_count = 0
                print(f"\n📝 Starting Report Generation Phase (0/{self.progress_callback.total_tasks['synthesis']} tasks)")
            
            # Technical compliance synthesis
            technical_result = await self.synthesis_chain.ainvoke({
                "input": json.dumps({
                    "findings": segment_results,
                    "standards": standards,
                    "synthesis_type": "technical",
                    "company_info": company_info
                }),
                "chat_history": []  # Start with empty chat history
            })
            technical_synthesis = technical_result.get("output", "")
            
            # Update synthesis progress
            if self.progress_callback:
                self.progress_callback.step_count += 1
                total = self.progress_callback.total_tasks["synthesis"]
                print(f"\r📝 Report Generation Progress: {self.progress_callback.step_count}/{total} tasks", end="")
                sys.stdout.flush()
            
            # Clear synthesis memory before next synthesis
            self.synthesis_memory.clear()
            
            # Business impact synthesis
            business_result = await self.synthesis_chain.ainvoke({
                "input": json.dumps({
                    "findings": segment_results,
                    "standards": standards,
                    "synthesis_type": "business",
                    "company_info": company_info
                }),
                "chat_history": []  # Start with empty chat history
            })
            business_synthesis = business_result.get("output", "")
            
            # Clear synthesis memory before final synthesis
            self.synthesis_memory.clear()
            
            # Final synthesis and report generation with synthesis chain
            final_result = await self.synthesis_chain.ainvoke({
                "input": json.dumps({
                    "technical_synthesis": technical_synthesis,
                    "business_synthesis": business_synthesis,
                    "standards": standards,
                    "company_info": company_info,
                    "output_format": "markdown"
                }),
                "chat_history": []  # Start with empty chat history
            })
            final_synthesis = final_result.get("output", "")
            
            # Clear all memories after processing
            self.analysis_memory.clear()
            self.synthesis_memory.clear()
            
            return {
                "success": True,
                "results": final_synthesis,
                "technical_analysis": technical_synthesis,
                "business_analysis": business_synthesis,
                "detailed_findings": segment_results
            }
            
        except Exception as e:
            # Clear memories even on error
            self.analysis_memory.clear()
            self.synthesis_memory.clear()
            
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}"
            }
            
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _segment_document(self, document: Dict) -> List[Dict]:
        """Split document into segments"""
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
