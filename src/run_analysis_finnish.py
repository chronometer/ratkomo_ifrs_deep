"""
Run IFRS analysis with Finnish output
"""
import asyncio
import sys
from pathlib import Path
from document_analyzer_finnish import OptimizedFinnishAnalyzer

async def main():
    if len(sys.argv) != 2:
        print("Käyttö: python run_analysis_finnish.py <tiedostopolku>")
        sys.exit(1)
    
    document_path = Path(sys.argv[1])
    if not document_path.exists():
        print(f"Virhe: Tiedostoa {document_path} ei löydy")
        sys.exit(1)
    
    analyzer = OptimizedFinnishAnalyzer()
    await analyzer.analyze_document(document_path)

if __name__ == "__main__":
    asyncio.run(main())
