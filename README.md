# CrewAI Agents with AWS Bedrock

This directory contains CrewAI agent orchestration framework configured to use AWS Bedrock with Amazon Nova Micro model.

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

- **Python 3.9+**
- **AWS Credentials**: Configured via `~/.aws/credentials` or environment variables
- **Bedrock Access**: IAM permissions for `bedrock:InvokeModel`

### Install Dependencies

First, install uv (fast Python package manager):

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH (or restart terminal)
export PATH="$HOME/.cargo/bin:$PATH"
```

Then install dependencies:

```bash
# Create virtual environment and install dependencies
uv sync

# Optional: Install full CrewAI (requires Rust compiler)
# uv add crewai
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

### Run AWS Bedrock Examples

#### Simple Bedrock Demo (Working! ✅)
Test AWS Bedrock Nova Micro directly:

```bash
# Using uv
uv run python simple_bedrock_example.py

# Or activate venv and run
source venv/bin/activate
python simple_bedrock_example.py
```

This demonstrates:
- **AWS Bedrock connectivity** with Nova Micro
- **Code generation** - Creates Python functions
- **Code review** - Analyzes generated code quality

#### Full CrewAI Example (Requires additional setup)
Execute the multi-agent example:

```bash
# Using uv
uv run python crew_example.py

# Or activate venv and run
source venv/bin/activate
python crew_example.py
```

Note: Full CrewAI requires additional dependencies and Rust compiler for tiktoken.

This demonstrates:
- **Coding Assistant**: Writes clean, efficient code
- **Code Reviewer**: Reviews for quality and best practices
- **Testing Agent**: Creates comprehensive tests

### Custom Agent Creation

Create your own agents:

```python
from crewai import Agent, Task, Crew
from langchain_aws import BedrockLLM

# Initialize Bedrock LLM
llm = BedrockLLM(
    model_id="amazon.nova-micro-v1:0",
    region_name="us-east-1"
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
    agent=agent
)

# Run crew
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()
```

## Advanced Configuration

### Custom Model Parameters

```python
llm = BedrockLLM(
    model_id="amazon.nova-micro-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0.7,      # Creativity (0.0-1.0)
        "max_tokens": 4096,      # Response length
        "top_p": 0.9,           # Nucleus sampling
        "top_k": 250,           # Top-k sampling
    }
)
```

### Multi-Agent Workflows

```python
# Create multiple agents
researcher = Agent(role="Researcher", goal="Gather information", llm=llm)
writer = Agent(role="Writer", goal="Create content", llm=llm)
editor = Agent(role="Editor", goal="Review and improve", llm=llm)

# Chain tasks
research_task = Task(description="Research topic", agent=researcher)
write_task = Task(description="Write article", agent=writer, context=[research_task])
edit_task = Task(description="Edit article", agent=editor, context=[write_task])

# Run crew
crew = Crew(agents=[researcher, writer, editor], tasks=[research_task, write_task, edit_task])
result = crew.kickoff()
```

## Performance & Cost Optimization

### Model Selection Guide

| Use Case | Recommended Model | Why |
|----------|------------------|-----|
| **Cost-Effective** | Nova Micro | Lowest cost, good for simple tasks |
| **Balanced** | Nova Lite | Good performance/cost ratio |
| **High Quality** | Nova Pro | Best quality, higher cost |
| **Complex Reasoning** | Claude 3.5 Sonnet | Excellent reasoning capabilities |

### Cost Monitoring

```bash
# Check Bedrock usage/costs
aws bedrock get-model-invocation-logging-config --region us-east-1
aws bedrock put-model-invocation-logging-config \
  --logging-config '{"cloudWatchConfig":{"logGroupName":"bedrock-logs"}}' \
  --region us-east-1
```

## Current Setup

- **Version**: CrewAI with AWS Bedrock integration
- **Model**: Amazon Nova Micro v1.0 (cost-optimized)
- **Region**: us-east-1 (configurable)
- **License**: MIT License

## Links

- **CrewAI Documentation**: https://docs.crewai.com
- **AWS Bedrock**: https://aws.amazon.com/bedrock/
- **Amazon Nova**: https://aws.amazon.com/ai/generative-ai/nova/
- **CrewAI GitHub**: https://github.com/crew-ai/crewai