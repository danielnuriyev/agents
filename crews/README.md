# Crews Directory Structure

This directory contains CrewAI crew configurations organized in YAML format.

## Structure

```
crews/
├── README.md
└── [crew_name]/
    ├── crew.[crew_name].yaml     # Main crew configuration
    ├── agent.[agent_name].yaml   # Individual agent configurations
    ├── task.[task_name].yaml     # Individual task configurations
    └── ...
```

Example:
```
crews/
├── README.md
└── code_development/
    ├── crew.code_development.yaml
    ├── agent.coding_assistant.yaml
    ├── agent.code_reviewer.yaml
    ├── agent.testing_agent.yaml
    ├── task.implement_task.yaml
    ├── task.review_task.yaml
    └── task.test_task.yaml
```

## File Formats

### crew.[crew_name].yaml
Defines the crew configuration with:
- `name`: Unique identifier for the crew
- `agents`: List of agent filenames (references to agent.*.yaml files)
- `tasks`: List of task filenames (references to task.*.yaml files)
- `verbose`: Enable/disable verbose output

### agent.[agent_name].yaml
Defines individual agent configurations with:
- `name`: Agent identifier
- `role`: Agent's role/title
- `goal`: Primary objective
- `backstory`: Detailed background/context
- `verbose`: Enable/disable verbose output
- `allow_delegation`: Whether agent can delegate tasks

### task.[task_name].yaml
Defines individual task configurations with:
- `name`: Task identifier
- `description`: Detailed task instructions
- `agent`: Reference to agent filename that should execute this task
- `expected_output`: Description of expected results
- `context`: Optional list of other task filenames this task depends on

## Example Crew: code_development

This example crew demonstrates a complete software development workflow with:
- **coding_assistant**: Writes clean, efficient code
- **code_reviewer**: Reviews code for quality and best practices
- **testing_agent**: Creates comprehensive test suites

Tasks are executed in sequence: implement → review → test.