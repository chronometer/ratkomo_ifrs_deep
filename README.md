# IFRS Document Analysis Framework

## Overview

The IFRS Document Analysis Framework is an advanced system that leverages Chain of Agents (CoA) architecture to analyze financial documents for IFRS compliance. Built using Langflow and Langchain for orchestration, and integrated with MCP servers for tooling, this framework provides a comprehensive solution for automated IFRS compliance checking.

## Features

- **Document Processing**: Automated parsing and analysis of financial statements
- **IFRS Compliance Analysis**: Rule-based compliance checking against IFRS standards
- **Chain of Agents Architecture**: Collaborative AI agents for comprehensive analysis
- **Visual Pipeline Construction**: Langflow integration for easy pipeline management
- **Extensible Tool Integration**: MCP server integration for custom tools
- **Comprehensive Reporting**: Detailed compliance reports with recommendations

## Architecture

The framework implements a Chain of Agents (CoA) architecture with three main agent types:

1. **Document Processor Agent**: Handles document ingestion and preprocessing
2. **Compliance Analyzer Agent**: Performs IFRS compliance analysis
3. **Manager Agent**: Synthesizes findings and generates reports

For detailed architecture information, see [Architecture Documentation](docs/architecture.md).

## Prerequisites

- Python 3.11+
- Langflow server
- MCP server
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-org/ifrs-analysis.git
cd ifrs-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

1. Start the Langflow server:
```bash
langflow start
```

2. Import the pipeline:
```bash
langflow import pipeline.json
```

3. Run the analysis:
```bash
python src/main.py --document financial_statement.pdf
```

## Documentation

- [Architecture Documentation](docs/architecture.md)
- [Technical Specification](docs/technical_spec.md)
- [API Documentation](docs/api.md)

## Development

For development setup and guidelines, see [Contributing Guidelines](CONTRIBUTING.md).

## Testing

Run the test suite:
```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support

For support and questions, please open an issue in the GitHub repository.
