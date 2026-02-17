#!/usr/bin/env python3
"""
CrewAI Example with AWS Bedrock Nova Micro

This script demonstrates how to use CrewAI with Amazon Nova Micro model
via AWS Bedrock for agent-based task completion.

Prerequisites:
- AWS credentials configured (via ~/.aws/credentials or environment variables)
- Required packages installed: pip install langchain-aws boto3 python-dotenv

Usage:
    python crew_example.py
"""

import os
from crewai import Agent, Task, Crew, LLM
from langchain_aws import ChatBedrock
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def create_bedrock_llm():
    """Create AWS Bedrock LLM instance for Nova Micro"""
    # Using the new CrewAI LLM class which is recommended in 1.x
    return LLM(
        model="bedrock/amazon.nova-micro-v1:0",
        temperature=0.7,
        max_tokens=4096,
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    )

def create_agents(llm):
    """Create CrewAI agents"""

    # Coding Assistant Agent
    coding_assistant = Agent(
        role="Senior Software Engineer",
        goal="Write clean, efficient, and well-documented code",
        backstory="""
        You are an experienced software engineer with 10+ years of experience.
        You excel at writing clean, maintainable code and following best practices.
        You always consider edge cases, error handling, and performance.
        """,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Code Reviewer Agent
    code_reviewer = Agent(
        role="Senior Code Reviewer",
        goal="Review code for quality, security, and best practices",
        backstory="""
        You are a meticulous code reviewer who catches bugs, security issues,
        and ensures code follows industry best practices. You provide constructive
        feedback and suggest improvements.
        """,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Testing Agent
    testing_agent = Agent(
        role="QA Engineer",
        goal="Create comprehensive tests and ensure code reliability",
        backstory="""
        You are a quality assurance expert who creates thorough test suites,
        identifies edge cases, and ensures code reliability and maintainability.
        """,
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    return coding_assistant, code_reviewer, testing_agent

def create_tasks(coding_assistant, code_reviewer, testing_agent):
    """Create CrewAI tasks"""

    # Task 1: Implement a feature
    implement_task = Task(
        description="""
        Implement a Python function that processes a list of dictionaries containing
        user data and returns aggregated statistics. The function should:

        1. Calculate average age of users
        2. Count users by country
        3. Find the most common interests
        4. Handle missing or invalid data gracefully

        Input format: List of dicts with keys: name, age, country, interests (list)
        Output format: Dict with aggregated statistics
        """,
        agent=coding_assistant,
        expected_output="Python function with comprehensive error handling and documentation"
    )

    # Task 2: Review the code
    review_task = Task(
        description="""
        Review the implemented function for:
        1. Code quality and readability
        2. Error handling and edge cases
        3. Performance considerations
        4. Security concerns
        5. Best practices compliance

        Provide specific recommendations for improvements.
        """,
        agent=code_reviewer,
        context=[implement_task],
        expected_output="Detailed code review with actionable recommendations"
    )

    # Task 3: Create tests
    test_task = Task(
        description="""
        Create comprehensive unit tests for the implemented function, including:
        1. Normal operation tests
        2. Edge cases (empty lists, missing data, invalid types)
        3. Error conditions
        4. Performance tests if applicable

        Use pytest framework and include test fixtures.
        """,
        agent=testing_agent,
        context=[implement_task],
        expected_output="Complete test suite with high coverage"
    )

    return implement_task, review_task, test_task

def main():
    """Main execution function"""
    print("🚀 Starting CrewAI with AWS Bedrock Nova Micro")
    print("=" * 50)

    try:
        # Initialize Bedrock LLM
        print("🤖 Initializing AWS Bedrock Nova Micro...")
        llm = create_bedrock_llm()

        # Create agents
        print("👥 Creating agents...")
        coding_assistant, code_reviewer, testing_agent = create_agents(llm)

        # Create tasks
        print("📋 Creating tasks...")
        implement_task, review_task, test_task = create_tasks(
            coding_assistant, code_reviewer, testing_agent
        )

        # Create and run crew
        print("⚡ Running crew...")
        crew = Crew(
            agents=[coding_assistant, code_reviewer, testing_agent],
            tasks=[implement_task, review_task, test_task],
            verbose=True
        )

        # Execute the crew
        result = crew.kickoff()

        print("\n" + "=" * 50)
        print("✅ Crew execution completed!")
        print("📄 Final Result:")
        print(result)

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check AWS credentials: aws configure")
        print("2. Verify AWS region has Bedrock access")
        print("3. Ensure Nova Micro model is available in your region")
        print("4. Check IAM permissions for bedrock:InvokeModel")

if __name__ == "__main__":
    main()