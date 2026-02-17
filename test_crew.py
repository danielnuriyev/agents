#!/usr/bin/env python3
"""
CrewAI Example with AWS Bedrock Nova Micro

This script demonstrates how to use CrewAI with Amazon Nova Micro model
via AWS Bedrock for agent-based task completion.

Prerequisites:
- AWS credentials configured (via ~/.aws/credentials or environment variables)
- Required packages installed: pip install langchain-aws boto3 python-dotenv

Usage:
    python test_crew.py [--output OUTPUT_FOLDER]

Arguments:
    --output OUTPUT_FOLDER    Folder to save results (default: crew_results)

Examples:
    python test_crew.py
    python test_crew.py --output my_experiment_1
    python test_crew.py --output ./results/run_2024_01_01
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from crewai import Agent, Task, Crew, LLM
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

def setup_output_folders(output_folder):
    """Create organized output folder structure"""
    base_path = Path(output_folder)

    # Create main output directory
    base_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different types of output
    console_dir = base_path / "console_output"
    results_dir = base_path / "results"
    metadata_dir = base_path / "metadata"

    console_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    return base_path, console_dir, results_dir, metadata_dir

def save_execution_metadata(metadata_dir, execution_info):
    """Save execution metadata to JSON file"""
    metadata_file = metadata_dir / "execution_metadata.json"

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(execution_info, f, indent=2, default=str)

def run_crew_with_output_capture(output_folder):
    """Run the crew and capture all output to files"""
    base_path, console_dir, results_dir, metadata_dir = setup_output_folders(output_folder)

    # Capture execution metadata
    execution_info = {
        "timestamp": datetime.now().isoformat(),
        "script": "test_crew.py",
        "output_folder": str(base_path),
        "python_version": sys.version,
        "crewai_version": "1.9.3",  # You might want to get this dynamically
        "model": "bedrock/amazon.nova-micro-v1:0",
        "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    }

    # Files for output
    console_log = console_dir / "crew_execution.log"
    final_result_file = results_dir / "final_result.txt"

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

        # Execute the crew and capture the result
        result = crew.kickoff()

        # Save final result
        with open(final_result_file, 'w', encoding='utf-8') as f:
            f.write(str(result))

        # Update execution info
        execution_info.update({
            "status": "success",
            "tasks_completed": 3,
            "agents_used": ["Senior Software Engineer", "Senior Code Reviewer", "QA Engineer"]
        })

        print(f"\n{'='*50}")
        print("✅ Crew execution completed!")
        print(f"📁 Results saved to: {base_path}")
        print(f"📄 Final result saved to: {final_result_file}")

        return True

    except Exception as e:
        error_msg = f"❌ Error: {e}"
        print(error_msg)

        # Save error to file
        error_file = results_dir / "error.log"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(error_msg)

        execution_info.update({
            "status": "error",
            "error": str(e)
        })

        print("\nTroubleshooting:")
        print("1. Check AWS credentials: aws configure")
        print("2. Verify AWS region has Bedrock access")
        print("3. Ensure Nova Micro model is available in your region")
        print("4. Check IAM permissions for bedrock:InvokeModel")

        return False

    finally:
        # Save metadata regardless of success/failure
        save_execution_metadata(metadata_dir, execution_info)

        # Copy console output to log file if it was captured
        # Note: This is a simplified approach. For full console capture,
        # you might want to redirect stdout at the script level
        try:
            with open(console_log, 'w', encoding='utf-8') as f:
                f.write("Console output captured during execution\n")
                f.write("Note: Full verbose output may not be captured in this log\n")
                f.write("Check the terminal output for complete verbose logs\n")
        except Exception:
            pass

def main():
    """Main execution function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Run CrewAI agents with AWS Bedrock Nova Micro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_crew.py
  python test_crew.py --output my_experiment_1
  python test_crew.py --output ./results/run_2024_01_01
        """
    )
    parser.add_argument(
        '--output', '-o',
        default='crew_results',
        help='Output folder for results (default: crew_results)'
    )

    args = parser.parse_args()

    print("🚀 Starting CrewAI with AWS Bedrock Nova Micro")
    print("=" * 50)
    print(f"📁 Output folder: {args.output}")

    success = run_crew_with_output_capture(args.output)

    if success:
        print(f"\n✅ All results saved to: {args.output}")
        print("📂 Folder structure:")
        print(f"   ├── console_output/crew_execution.log")
        print(f"   ├── results/final_result.txt")
        print(f"   └── metadata/execution_metadata.json")
    else:
        print(f"\n❌ Execution failed. Check error logs in: {args.output}/results/error.log")

if __name__ == "__main__":
    main()