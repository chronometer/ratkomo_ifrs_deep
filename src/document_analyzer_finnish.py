"""
Finnish version of the optimized document analyzer for IFRS documents
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import os
from dotenv import load_dotenv
import logging
from dataclasses import dataclass
from langchain.schema import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.models.openai import OpenAIModel, ModelRequestParameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentSegment(BaseModel):
    """Represents a segment of the document with metadata"""
    content: str = Field(..., description="The text content of the segment")
    page_number: Optional[int] = Field(None, description="Page number where the segment appears")
    segment_type: Optional[str] = Field(None, description="Type of content in the segment")

class AnalysisResult(BaseModel):
    """Structured analysis result with IFRS compliance details"""
    segments: List[DocumentSegment]
    key_findings: List[str]
    compliance_status: str
    relevant_standards: List[Dict[str, str]]
    recommendations: List[str]

class OpenRouterModel(OpenAIModel):
    def __init__(self, model_name, base_url=None, api_key=None):
        super().__init__(model_name, base_url=base_url, api_key=api_key)
        self._api_key = api_key
        self._base_url = base_url
        self._client = None
    
    async def request(self, messages, settings=None, parameters=None):
        if not self._client:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                default_headers={
                    'HTTP-Referer': 'http://localhost:8000',
                    'X-Title': 'Document Analyzer'
                }
            )
        
        try:
            response = await self._client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    'role': 'system' if m.type == 'system' else 'user',
                    'content': m.content
                } for m in messages],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in OpenRouter request: {e}")
            return None

class OptimizedFinnishAnalyzer:
    def __init__(self):
        """Initialize the optimized document analyzer"""
        load_dotenv()
        
        self.model = OpenRouterModel(
            model_name=os.getenv('DEFAULT_LLM_MODEL'),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        
        self.min_segment_size = 1000
        self.max_segment_size = 4000
        self.batch_size = 3

    def _get_finnish_system_prompt(self):
        return '''Olet IFRS-asiantuntija, joka analysoi taloudellisia dokumentteja.
            Analysoi seuraavat dokumenttiosiot ja anna kattava analyysi, joka sisältää:
            1. Keskeiset löydökset kunkin IFRS-standardin osalta
            2. Tarkat standardiviittaukset (esim. IFRS 15.31)
            3. Selkeät parannusehdotukset
            4. Yleinen vaatimustenmukaisuuden tila
            
            Keskity erityisesti:
            - Tulojen kirjaamiseen (IFRS 15)
            - Rahoitusinstrumentteihin (IFRS 9)
            - Vuokrajärjestelyihin (IFRS 16)
            - Segmenttiraportointiin (IFRS 8)'''

    async def analyze_batch(self, segments: List[DocumentSegment]):
        """Analyze a batch of segments in a single API call"""
        messages = [
            SystemMessage(content=self._get_finnish_system_prompt()),
            HumanMessage(content=f"Analysoi nämä dokumenttiosiot:\n\n" + "\n\n".join(
                f"[Type: {s.segment_type}]\n{s.content}" for s in segments
            ))
        ]
        
        logging.info(f"Sending request with messages: {messages}")
        response = await self.model.request(messages)
        return self._process_analysis_response(response, segments)

    def _process_analysis_response(self, response: str, segments: List[DocumentSegment]) -> AnalysisResult:
        """Process the API response into a structured analysis result"""
        return AnalysisResult(
            segments=segments,
            key_findings=self._extract_findings(response),
            compliance_status=self._extract_compliance_status(response),
            relevant_standards=self._extract_standards(response),
            recommendations=self._extract_recommendations(response)
        )

    def _generate_report(self, results: List[AnalysisResult]) -> str:
        """Generate final analysis report in Finnish markdown format"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# IFRS-vaatimustenmukaisuuden analyysi

Luotu: {now}

## Yhteenveto
{self._generate_executive_summary(results)}

## Keskeiset löydökset
{self._generate_detailed_analysis(results)}

## Suositukset
"""
        
        for result in results:
            for recommendation in result.recommendations:
                report += f"- {recommendation}\n"
        
        return report

    def _generate_executive_summary(self, results: List[AnalysisResult]) -> str:
        """Generate executive summary from all results in Finnish"""
        summary = ""
        for result in results:
            summary += f"- Vaatimustenmukaisuuden tila: {result.compliance_status}\n"
            for finding in result.key_findings[:3]:  # Top 3 findings
                summary += f"- {finding}\n"
        return summary

    def _generate_detailed_analysis(self, results: List[AnalysisResult]) -> str:
        """Generate detailed analysis section in Finnish"""
        analysis = ""
        for result in results:
            analysis += "\n### Standardikohtainen analyysi\n"
            for standard in result.relevant_standards:
                for std_name, details in standard.items():
                    analysis += f"\n#### {std_name}\n{details}\n"
        return analysis

    async def analyze_document(self, document_path: Path):
        """Analyze a document and generate a Finnish report"""
        logging.info("Aloitetaan optimoitu IFRS-analyysi...")
        
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        segments = self.segment_document(content)
        logging.info(f"Dokumentti jaettu {len(segments)} optimoituun osioon")
        
        results = []
        for i in range(0, len(segments), self.batch_size):
            batch = segments[i:i + self.batch_size]
            result = await self.analyze_batch(batch)
            results.append(result)
            logging.info(f"Käsitelty erä {i//self.batch_size + 1}/{(len(segments) + self.batch_size - 1)//self.batch_size}")
        
        report = self._generate_report(results)
        
        output_path = Path('output/analyysi_raportti.md')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding='utf-8')
        
        logging.info(f"Analyysi valmis. Raportti tallennettu tiedostoon {output_path}")
        return report

    def segment_document(self, content: str) -> List[DocumentSegment]:
        """Smart document segmentation that combines related content"""
        lines = content.split('\n')
        segments = []
        current_segment = []
        current_size = 0
        
        for line in lines:
            line_size = len(line)
            if current_size + line_size > self.max_segment_size and current_segment:
                segments.append(DocumentSegment(
                    content='\n'.join(current_segment),
                    segment_type=self._detect_segment_type('\n'.join(current_segment[:3]))
                ))
                current_segment = []
                current_size = 0
            
            current_segment.append(line)
            current_size += line_size
        
        if current_segment:
            segments.append(DocumentSegment(
                content='\n'.join(current_segment),
                segment_type=self._detect_segment_type('\n'.join(current_segment[:3]))
            ))
        
        return segments

    def _detect_segment_type(self, title: str) -> str:
        """Detect the type of content in a segment based on its title"""
        title = title.lower()
        if any(word in title for word in ['tulos', 'liikevaihto', 'tase']):
            return "financial"
        elif any(word in title for word in ['strategia', 'visio']):
            return "strategic"
        elif any(word in title for word in ['riski', 'valvonta']):
            return "risk"
        return "general"

    def _extract_findings(self, content: str) -> List[str]:
        """Extract key findings from analysis response"""
        findings = []
        for line in content.split('\n'):
            if line.strip().startswith(('- ', '* ')) and not line.startswith(('- Suositus', '* Suositus')):
                findings.append(line.strip()[2:])
        return findings

    def _extract_compliance_status(self, content: str) -> str:
        """Extract overall compliance status"""
        for line in content.split('\n'):
            if "vaatimustenmukaisuuden tila" in line.lower():
                return line.split(':')[-1].strip()
        return "Ei määritelty"

    def _extract_standards(self, content: str) -> List[Dict[str, str]]:
        """Extract relevant IFRS standards"""
        standards = []
        current_standard = None
        current_details = []
        
        for line in content.split('\n'):
            if "IFRS" in line and ":" in line:
                if current_standard:
                    standards.append({current_standard: '\n'.join(current_details)})
                current_standard = line.split(':')[0].strip()
                current_details = [line.split(':')[1].strip()]
            elif current_standard and line.strip():
                current_details.append(line.strip())
        
        if current_standard:
            standards.append({current_standard: '\n'.join(current_details)})
        
        return standards

    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract recommendations from analysis"""
        recommendations = []
        for line in content.split('\n'):
            if any(word in line.lower() for word in ['suositus', 'suositellaan', 'tulisi']):
                recommendations.append(line.strip())
        return recommendations
