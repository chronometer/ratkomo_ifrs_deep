# Testing Documentation

## Overview

This directory contains the test suite for the IFRS Document Analysis Framework. The tests are organized into different categories and use pytest as the testing framework.

## Test Structure

```
tests/
├── unit/               # Unit tests
│   └── agents/        # Tests for individual agents
├── integration/        # Integration tests
├── e2e/               # End-to-end tests
├── performance/       # Performance tests
├── fixtures/          # Test fixtures and data
└── requirements-test.txt  # Test dependencies
```

## Test Categories

### Unit Tests
- `test_base_agent.py`: Tests for the base agent functionality
- `test_worker_agent.py`: Tests for worker agent implementation
- `test_manager_agent.py`: Tests for manager agent implementation
- `test_chain_of_agents.py`: Tests for chain orchestration

### Test Markers
- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.performance`: Performance tests

## Running Tests

### Prerequisites
1. Install test dependencies:
```bash
pip install -r tests/requirements-test.txt
```

### Running All Tests
From the project root:
```bash
./run_tests.sh
```

### Running Specific Test Categories
```bash
# Run only unit tests
pytest tests/unit -v

# Run only integration tests
pytest tests/integration -v

# Run tests with specific marker
pytest -m unit -v
```

### Coverage Reports
The test runner generates coverage reports in multiple formats:
- Terminal output with missing lines
- HTML report in `coverage_html/`
- XML report in `coverage.xml`

## Writing Tests

### Test Guidelines
1. Use appropriate markers for test categorization
2. Include docstrings explaining test purpose
3. Use fixtures for common test data
4. Mock external dependencies
5. Test both success and error cases
6. Include assertions for all relevant conditions

### Example Test Structure
```python
@pytest.mark.unit
class TestComponent:
    """Test cases for Component"""
    
    def test_success_case(self):
        """Test successful operation"""
        # Arrange
        component = Component()
        
        # Act
        result = component.operation()
        
        # Assert
        assert result.success
    
    def test_error_case(self):
        """Test error handling"""
        # Arrange
        component = Component()
        
        # Act & Assert
        with pytest.raises(ValueError):
            component.invalid_operation()
```

### Fixtures
Common test fixtures are available in the fixtures directory:
- `test_context`: Chain context for testing
- `test_document`: Sample document data
- `mock_llm_response`: Mocked LLM responses

## CI/CD Integration
The test suite is integrated with CI/CD pipelines:
- Tests run on every pull request
- Coverage reports are uploaded to the CI platform
- Failed tests block merging

## Performance Testing
Performance tests measure:
- Processing time per segment
- Memory usage
- Worker agent throughput
- Chain completion time

Run performance tests separately:
```bash
pytest tests/performance -v
```

## Troubleshooting

### Common Issues
1. **Import Errors**
   - Ensure PYTHONPATH includes project root
   - Check for circular imports

2. **Async Test Failures**
   - Use `@pytest.mark.asyncio` for async tests
   - Ensure proper test isolation

3. **Coverage Issues**
   - Check .coveragerc configuration
   - Ensure all source files are included

### Debug Mode
Run tests with debug logging:
```bash
pytest -v --log-cli-level=DEBUG
