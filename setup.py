from setuptools import setup, find_packages

setup(
    name="ifrs-document-analysis",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "langchain==0.0.176",
        "langflow==0.0.78",
        "pydantic>=1.0.0,<2.0.0",
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "pytest-mock",
            "pytest-env",
            "pytest-xdist",
            "pytest-timeout",
            "coverage",
        ],
    },
    author="Ratkomo",
    description="IFRS Document Analysis Framework using Chain of Agents",
    keywords="ifrs, document-analysis, chain-of-agents, llm",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Accounting",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
