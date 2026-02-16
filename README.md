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

### Run CrewAI Example

Execute the multi-agent example:

```bash
# Using uv
uv run python crew_example.py

# Or activate venv and run
source venv/bin/activate
python crew_example.py
```

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

## Available Bedrock Models

CrewAI supports these AWS Bedrock models:

### Amazon Nova Models
- `amazon.nova-micro-v1:0` ⭐ **Recommended for cost-efficiency**
- `amazon.nova-lite-v1:0`
- `amazon.nova-pro-v1:0`

### Anthropic Claude Models
- `anthropic.claude-3-5-sonnet-20240620-v1:0`
- `anthropic.claude-3-haiku-20240307-v1:0`
- `anthropic.claude-3-sonnet-20240229-v1:0`

### Meta Llama Models
- `meta.llama3-8b-instruct-v1:0`
- `meta.llama3-70b-instruct-v1:0`

## Project Structure

```
crewai-agents/
├── venv/                    # Python virtual environment (created by uv)
├── pyproject.toml          # uv project configuration
├── .env.example            # Environment variables template
├── crew_example.py         # Multi-agent CrewAI example
├── LICENSE                 # MIT License
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# AWS Configuration
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_DEFAULT_REGION=us-east-1

# Optional: Use specific profile
AWS_PROFILE=default

# CrewAI Configuration
CREWAI_LOG_LEVEL=INFO
```

## Troubleshooting

### Common Issues

#### 1. AWS Credentials Error
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```
**Solution:**
```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
```

#### 2. Bedrock Model Not Available
```
ValidationException: The model is not available in us-east-1
```
**Solution:**
- Check model availability in your region
- Try a different AWS region
- Use a different model (e.g., Claude instead of Nova)

#### 3. IAM Permissions Error
```
AccessDeniedException: User is not authorized
```
**Solution:**
- Add Bedrock permissions to your IAM user/role
- Ensure the policy allows `bedrock:InvokeModel`

#### 4. CrewAI Installation Issues
```
ERROR: Failed building wheel for tiktoken
```
**Solution:**
- Install Rust compiler: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Or use the simplified setup without full CrewAI

### Debug Commands

```bash
# Test AWS credentials
aws sts get-caller-identity

# List available Bedrock models
aws bedrock list-foundation-models --region us-east-1

# Test Bedrock invocation
aws bedrock-runtime invoke-model \
  --model-id amazon.nova-micro-v1:0 \
  --body '{"prompt": "Hello"}' \
  --region us-east-1 \
  output.json
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