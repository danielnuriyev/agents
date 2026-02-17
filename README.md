# CrewAI Agents with AWS Bedrock

This directory contains a CrewAI agent orchestration framework configured to use AWS Bedrock with the Amazon Nova Micro model.

## What is CrewAI?

CrewAI is a framework for orchestrating AI agents to work together on complex tasks. Unlike single-agent systems, CrewAI enables multiple agents with different roles and expertise to collaborate and complete sophisticated workflows.

## Key Features

- **Multi-Agent Orchestration**: Coordinate multiple AI agents with different roles
- **AWS Bedrock Integration**: Use Amazon Nova Micro and other Bedrock models
- **Task-Based Workflows**: Break down complex tasks into manageable steps
- **Agent Collaboration**: Agents can delegate tasks and share context
- **Python-Native**: Full programmatic control with Python

## Installation

### Prerequisites

- **uv**: Fast Python package manager (recommended)
- **Python 3.10 - 3.13** (Note: Python 3.14 is currently NOT supported by CrewAI)
- **AWS Credentials**: Configured via `~/.aws/credentials` or environment variables
- **Bedrock Access**: IAM permissions for `bedrock:InvokeModel`

### Install and Run with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that simplifies environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies and create virtual environment
uv sync

# Run the crew using uv run (automatically handles the environment)
uv run run_crew.py --crew ./crews/example
```

### Manual Installation (Alternative)

It is recommended to use a virtual environment:

```bash
# Create virtual environment with a supported Python version (3.13 recommended)
python3.13 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install crewai langchain-aws boto3 python-dotenv pyyaml
```

## AWS Bedrock Configuration

### 1. AWS Credentials

Your AWS credentials should be configured. The system will automatically use:
- `~/.aws/credentials` file
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- AWS profile (if specified)

### 2. AWS Region

Set your preferred AWS region:

```bash
export AWS_DEFAULT_REGION=us-east-1
# Or add to .env file
```

### 3. IAM Permissions

Ensure your AWS user/role has these Bedrock permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream"
            ],
            "Resource": "*"
        }
    ]
}
```

## Usage

### Run CrewAI with YAML Configurations (New! 🚀)

The system now supports loading agents, tasks, and crew configurations from YAML files.
**Agents actually write and execute code** to verify their work!

```bash
# Using uv (Recommended)
uv run run_crew.py --crew ./crews/example

# Using standard Python
python run_crew.py --crew ./crews/example
```

**Arguments:**
- `--crew CREW_FOLDER`: Path to the folder containing YAML configurations (required)
- `--output OUTPUT_FOLDER`: Folder to save results (default: `./.results/crew-{timestamp}`)

**Examples:**
```bash
# Run the example crew
uv run run_crew.py --crew ./crews/example

# Specify custom output folder
uv run run_crew.py --crew ./crews/example --output my_experiment_1
```

**Output Structure:**
Results are automatically saved to a timestamped directory in `.results/`:
```
.results/
└── crew-20241201143025/          # Timestamped run folder
    ├── crew_execution.log        # Console logs
    ├── final_result.md           # Complete analysis & test results
    ├── execution_metadata.json   # Metadata
    ├── find_max_even.py              # Main function
    ├── test_validation.py            # Test functions (run concurrently!)
    ├── test_performance.py           # Performance benchmarks
    ├── test_security.py              # Security vulnerability tests
    ├── test_execution_results.json   # Concurrent execution results with timing
    └── ...                           # All extracted functions
```

**✨ Key Features:**
- **Code Extraction**: Each function is automatically extracted into separate `.py` files
- **Concurrent Test Execution**: Multiple test functions run in parallel using subprocesses
- **Individual Files**: Get `find_max_even.py`, `test_validation.py`, etc. as separate files
- **Real Execution Results**: Tests actually run and report PASS/FAIL with timing data
- **Performance**: Concurrent execution provides ~3x speedup for multiple tests
- **Test Analysis Agent**: Reviews all test results and provides fix recommendations

### Legacy Examples

#### Simple Bedrock Demo (Working! ✅)
Test AWS Bedrock Nova Micro directly using `boto3`:

```bash
# Using uv
uv run test_bedrock.py

# Using standard Python
source .venv/bin/activate
python test_bedrock.py
```

This demonstrates:
- **AWS Bedrock connectivity** with Nova Micro
- **Direct API usage** - Efficiently invokes the model
- **Code generation & review** - Creates and analyzes Python code

#### Full CrewAI Example (test_crew.py)
Execute the hardcoded multi-agent orchestration example:

```bash
# Using uv
uv run test_crew.py

# Using standard Python
source .venv/bin/activate
python test_crew.py [--output OUTPUT_FOLDER]
```

**Arguments:**
- `--output OUTPUT_FOLDER`: Folder to save results (default: `crew_results`)

**Examples:**
```bash
# Use default output folder
uv run test_crew.py

# Specify custom output folder
uv run test_crew.py --output my_experiment_1
```

This demonstrates:
- **Coding Assistant**: Writes clean, efficient code
- **Code Reviewer**: Reviews for quality and best practices
- **Testing Agent**: Creates comprehensive tests
- **Collaborative Workflow**: Agents working together to complete a complex task

**Output Structure:**
```
OUTPUT_FOLDER/
├── crew_execution.log          # Console logs and verbose output
├── final_result.md             # Final crew execution result
├── error.log                   # Error logs (if any)
└── execution_metadata.json     # Execution info and metadata
```

### Custom Agent Creation

Create your own agents using the CrewAI `LLM` class:

```python
from crewai import Agent, Task, Crew, LLM
import os

# Initialize Bedrock LLM using CrewAI's LLM class
llm = LLM(
    model="bedrock/amazon.nova-micro-v1:0",
    temperature=0.7,
    max_tokens=4096,
    region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
)

# Create custom agent
agent = Agent(
    role="Data Analyst",
    goal="Analyze data and provide insights",
    backstory="Expert data analyst with 10 years experience",
    llm=llm,
    verbose=True
)

# Create task
task = Task(
    description="Analyze sales data from Q1",
    agent=agent,
    expected_output="A summary report of Q1 sales trends"
)

# Run crew
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

## Advanced Configuration

### Custom Model Parameters

```python
llm = LLM(
    model="bedrock/amazon.nova-micro-v1:0",
    temperature=0.5,      # More focused (0.0-1.0)
    max_tokens=2048,     # Response length limit
    # Additional Bedrock parameters can be passed if needed
)
```

### Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Cost-Effective** | `bedrock/amazon.nova-micro-v1:0` | Lowest cost, extremely fast |
| **Balanced** | `bedrock/amazon.nova-lite-v1:0` | Good performance/cost ratio |
| **High Quality** | `bedrock/amazon.nova-pro-v1:0` | Best quality for complex tasks |
| **Industry Leading** | `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0` | Top-tier reasoning and coding |

## Current Setup

- **Framework**: CrewAI 1.9.3+
- **Primary Model**: Amazon Nova Micro (via AWS Bedrock)
- **Environment**: Python 3.13 (Python 3.14 unsupported)
- **License**: MIT License

## Links

- **CrewAI Documentation**: https://docs.crewai.com
- **AWS Bedrock Documentation**: https://docs.aws.amazon.com/bedrock/
- **Amazon Nova Guide**: https://docs.aws.amazon.com/nova/latest/userguide/
