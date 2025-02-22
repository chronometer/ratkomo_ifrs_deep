#!/bin/bash

# Install test requirements
pip install -r tests/requirements-test.txt

# Run tests with coverage
pytest tests/ \
    --cov=src \
    --cov-config=.coveragerc \
    --cov-report=term-missing \
    --cov-report=html \
    --cov-report=xml \
    -v \
    --asyncio-mode=auto \
    --tb=short \
    "$@"

# Open coverage report if tests pass
if [ $? -eq 0 ]; then
    echo "Tests passed! Opening coverage report..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open coverage_html/index.html
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open coverage_html/index.html
    elif [[ "$OSTYPE" == "msys" ]]; then
        start coverage_html/index.html
    fi
fi
