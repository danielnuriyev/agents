#!/usr/bin/env python3
"""
CrewAI Runner with AWS Bedrock Nova Micro

This script loads CrewAI agents, tasks, and crew configurations from YAML files
in a specified folder and executes them using Amazon Nova Micro model via AWS Bedrock.

Prerequisites:
- AWS credentials configured (via ~/.aws/credentials or environment variables)
- Required packages installed: pip install langchain-aws boto3 python-dotenv pyyaml
- Crew configuration folder with YAML files (see crews/example/ for structure)

Usage:
    python run_crew.py --crew CREW_FOLDER [--output OUTPUT_FOLDER]

Arguments:
    --crew CREW_FOLDER        Path to crew configuration folder (required)
    --output OUTPUT_FOLDER     Folder to save results (default: ./.results/crew-{yyyymmddhhmmss})

Examples:
    python run_crew.py --crew ./crews/example
    python run_crew.py --crew ./crews/example --output my_experiment_1
    python run_crew.py --crew ./crews/example --output ./results/run_2024_01_01
"""

import os
import sys
import json
import argparse
import yaml
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

def load_agents_from_folder(crew_folder, llm):
    """Load agent configurations from YAML files in the specified crew folder"""
    agents = {}
    agents_dir = Path(crew_folder)

    # Find all agent YAML files
    agent_files = agents_dir.glob("agent.*.yaml")

    for agent_file in agent_files:
        with open(agent_file, 'r', encoding='utf-8') as f:
            agent_config = yaml.safe_load(f)

        agent = Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            llm=llm,
            verbose=agent_config.get('verbose', True),
            allow_delegation=agent_config.get('allow_delegation', False)
        )

        # Use the filename (without extension) as the key
        agent_key = agent_file.stem  # e.g., "agent.coding_assistant" -> "agent.coding_assistant"
        agents[agent_key] = agent

    return agents

def load_tasks_from_folder(crew_folder, agents):
    """Load task configurations from YAML files in the specified crew folder"""
    tasks = {}
    tasks_dir = Path(crew_folder)

    # Find all task YAML files
    task_files = sorted(list(tasks_dir.glob("task.*.yaml")))

    # First pass: load all tasks without context
    for task_file in task_files:
        with open(task_file, 'r', encoding='utf-8') as f:
            task_config = yaml.safe_load(f)

        # Get the agent for this task
        agent_key = task_config['agent']
        if agent_key not in agents:
            raise ValueError(f"Agent '{agent_key}' not found for task '{task_config['name']}'")

        agent = agents[agent_key]

        # Use the filename (without extension) as the key
        task_key = task_file.stem
        
        # Store config for second pass
        tasks[task_key] = {
            'config': task_config,
            'agent': agent,
            'instance': None
        }

    # Second pass: instantiate tasks with context
    for task_key, task_data in tasks.items():
        task_config = task_data['config']
        
        # Build task kwargs
        task_kwargs = {
            'description': task_config['description'],
            'agent': task_data['agent'],
            'expected_output': task_config['expected_output']
        }

        # Add context if specified
        if 'context' in task_config:
            context_tasks = []
            for context_task_key in task_config['context']:
                if context_task_key not in tasks:
                    raise ValueError(f"Context task '{context_task_key}' not found for task '{task_config['name']}'")
                
                # We need the instance, but it might not be created yet if there are cycles or complex deps
                # However, for simple linear deps, we can just ensure we process in order or recursively
                # For now, let's assume we need to instantiate them
                if tasks[context_task_key]['instance'] is None:
                    # This is a simple recursive instantiation
                    _instantiate_task(context_task_key, tasks)
                
                context_tasks.append(tasks[context_task_key]['instance'])
            task_kwargs['context'] = context_tasks

        if tasks[task_key]['instance'] is None:
            tasks[task_key]['instance'] = Task(**task_kwargs)

    return {k: v['instance'] for k, v in tasks.items()}

def _instantiate_task(task_key, tasks):
    """Helper to recursively instantiate tasks with their context"""
    if tasks[task_key]['instance'] is not None:
        return tasks[task_key]['instance']
    
    task_data = tasks[task_key]
    task_config = task_data['config']
    
    task_kwargs = {
        'description': task_config['description'],
        'agent': task_data['agent'],
        'expected_output': task_config['expected_output']
    }
    
    if 'context' in task_config:
        context_tasks = []
        for context_task_key in task_config['context']:
            if context_task_key not in tasks:
                raise ValueError(f"Context task '{context_task_key}' not found")
            context_tasks.append(_instantiate_task(context_task_key, tasks))
        task_kwargs['context'] = context_tasks
        
    tasks[task_key]['instance'] = Task(**task_kwargs)
    return tasks[task_key]['instance']

def load_crew_from_folder(crew_folder, agents, tasks):
    """Load crew configuration from YAML file in the specified crew folder"""
    crew_dir = Path(crew_folder)
    crew_file = crew_dir / f"crew.{crew_dir.name}.yaml"

    if not crew_file.exists():
        raise FileNotFoundError(f"Crew configuration file not found: {crew_file}")

    with open(crew_file, 'r', encoding='utf-8') as f:
        crew_config = yaml.safe_load(f)

    # Get the agents and tasks for this crew
    crew_agents = []
    for agent_key in crew_config['agents']:
        if agent_key not in agents:
            raise ValueError(f"Agent '{agent_key}' not found in crew configuration")
        crew_agents.append(agents[agent_key])

    crew_tasks = []
    for task_key in crew_config['tasks']:
        if task_key not in tasks:
            raise ValueError(f"Task '{task_key}' not found in crew configuration")
        crew_tasks.append(tasks[task_key])

    crew = Crew(
        agents=crew_agents,
        tasks=crew_tasks,
        verbose=crew_config.get('verbose', True)
    )

    return crew

def setup_output_folders(output_folder):
    """Create output folder structure"""
    base_path = Path(output_folder)

    # Create main output directory
    base_path.mkdir(parents=True, exist_ok=True)

    return base_path, base_path, base_path, base_path

def save_execution_metadata(metadata_dir, execution_info):
    """Save execution metadata to JSON file"""
    metadata_file = metadata_dir / "execution_metadata.json"

    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(execution_info, f, indent=2, default=str)

def extract_code_blocks(output_dir, final_result_file):
    """Extract the final version of code blocks from markdown and save each function as separate files"""
    import re
    import ast

    # Language to file extension mapping
    extensions = {
        'python': '.py',
        'py': '.py',
        'javascript': '.js',
        'js': '.js',
        'typescript': '.ts',
        'java': '.java',
        'cpp': '.cpp',
        'c++': '.cpp',
        'c': '.c',
        'go': '.go',
        'rust': '.rs',
        'ruby': '.rb',
        'php': '.php',
        'html': '.html',
        'css': '.css',
        'sql': '.sql',
        'bash': '.sh',
        'shell': '.sh',
        'yaml': '.yaml',
        'yml': '.yml',
        'json': '.json',
        'xml': '.xml',
        'markdown': '.md',
        'md': '.md'
    }

    # Read the final result markdown file
    with open(final_result_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract code blocks using regex
    code_block_pattern = r'```(\w+)?\n(.*?)\n```'
    code_blocks = re.findall(code_block_pattern, content, re.DOTALL)

    # Group code blocks by language and keep only the last one for each language
    final_code_blocks = {}
    for lang, code in code_blocks:
        lang = lang.lower() if lang else 'txt'
        final_code_blocks[lang] = code.strip()

    extracted_files = []

    for lang, code in final_code_blocks.items():
        ext = extensions.get(lang, '.txt')

        if lang == 'python' and code.strip():
            # For Python, extract each function into separate files
            extracted_files.extend(extract_python_functions(code, output_dir, ext))
        else:
            # For other languages, save as single file
            if lang and lang != 'txt':
                filename = f"final_code_{lang}{ext}"
            else:
                filename = f"final_code.txt"

            code_file = output_dir / filename
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(code)
            extracted_files.append(str(code_file))

    return extracted_files

def extract_python_functions(code, output_dir, ext):
    """Extract each Python function from code and save to separate files"""
    import re

    extracted_files = []

    # Split code by function definitions using regex
    # This pattern matches function definitions and captures everything up to the next function or end
    function_pattern = r'(def\s+\w+\s*\([^)]*\)\s*:.*?)(?=\n\s*def\s|\n\s*class\s|\Z)'
    functions = re.findall(function_pattern, code, re.DOTALL)

    # If regex doesn't work well, fall back to simple line-based splitting
    if not functions:
        # Simple fallback: split by 'def ' and 'class ' lines
        parts = re.split(r'(?=^def\s|^class\s)', code.strip(), flags=re.MULTILINE)
        functions = [part for part in parts if part.strip() and (part.strip().startswith('def ') or part.strip().startswith('class '))]

    # Process each function/class
    for func_code in functions:
        func_code = func_code.strip()
        if not func_code:
            continue

        # Extract function/class name
        first_line = func_code.split('\n')[0].strip()
        if first_line.startswith('def '):
            name_match = re.match(r'def\s+(\w+)', first_line)
        elif first_line.startswith('class '):
            name_match = re.match(r'class\s+(\w+)', first_line)
        else:
            continue

        if name_match:
            func_name = name_match.group(1)

            # Save to file
            filename = f"{func_name}{ext}"
            code_file = output_dir / filename
            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(func_code)
            extracted_files.append(str(code_file))

    # If no functions found, save entire code as fallback
    if not extracted_files:
        filename = f"final_code_python{ext}"
        code_file = output_dir / filename
        with open(code_file, 'w', encoding='utf-8') as f:
            f.write(code)
        extracted_files.append(str(code_file))

    return extracted_files

def run_single_test(test_file, output_dir):
    """Run a single test file and return results"""
    import subprocess
    import sys
    import time

    test_name = test_file.stem
    start_time = time.time()

    # Create a simple runner that loads the main function and runs the test
    runner_script = f'''
import sys
import os
import traceback

# Change to the output directory
os.chdir(r"{str(output_dir)}")

# Try to load the main function
try:
    # Execute the main function file
    with open('find_max_even.py', 'r') as f:
        exec(f.read())

    if 'find_max_even' not in globals():
        print("ERROR: find_max_even function not found after loading")
        sys.exit(1)

    print("Successfully loaded find_max_even function")

except Exception as e:
    print(f"ERROR loading main function: {{e}}")
    traceback.print_exc()
    sys.exit(1)

# Execute the test file
try:
    with open(r"{str(test_file.name)}", 'r') as f:
        exec(f.read())
    print("Test completed successfully")

except Exception as e:
    print(f"ERROR during test execution: {{e}}")
    traceback.print_exc()
    sys.exit(1)
'''

    runner_filename = f"run_{test_name}.py"
    runner_path = output_dir / runner_filename
    with open(runner_path, 'w') as f:
        f.write(runner_script)

    try:
        result = subprocess.run(
            [sys.executable, runner_filename],
            capture_output=True,
            text=True,
            timeout=15,  # 15 second timeout
            cwd=str(output_dir)
        )

        execution_time = time.time() - start_time

        test_result = {
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'success': result.returncode == 0,
            'execution_time': f"{execution_time:.3f}s"
        }

    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        test_result = {
            'return_code': -1,
            'stdout': '',
            'stderr': 'Test execution timed out after 15 seconds',
            'success': False,
            'execution_time': f"{execution_time:.3f}s (TIMEOUT)"
        }
    except Exception as e:
        execution_time = time.time() - start_time
        test_result = {
            'return_code': -1,
            'stdout': '',
            'stderr': f'Execution setup error: {str(e)}',
            'success': False,
            'execution_time': f"{execution_time:.3f}s (ERROR)"
        }

    # Clean up runner file
    try:
        runner_path.unlink()
    except:
        pass

    return test_name, test_result

def run_extracted_tests(output_dir):
    """Execute extracted test functions concurrently and capture results"""
    import concurrent.futures
    import time

    # Find all Python files that look like test functions
    test_files = []
    for file_path in output_dir.glob("*.py"):
        filename = file_path.name
        # Look for files that start with "test_"
        if filename.startswith("test_"):
            test_files.append(file_path)

    if not test_files:
        return {"error": "No test files found"}

    print(f"Running {len(test_files)} tests concurrently...")

    # Execute tests concurrently using ThreadPoolExecutor
    all_results = {}
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(test_files), 8)) as executor:
        # Submit all test executions
        future_to_test = {
            executor.submit(run_single_test, test_file, output_dir): test_file
            for test_file in test_files
        }

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_test):
            test_name, result = future.result()
            all_results[test_name] = result
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"  {test_name}: {status} ({result['execution_time']})")

    total_time = time.time() - start_time
    print(".3f")

    # Add summary to results
    passed = sum(1 for r in all_results.values() if r['success'])
    total = len(all_results)
    all_results['_summary'] = {
        'total_tests': total,
        'passed': passed,
        'failed': total - passed,
        'total_execution_time': f"{total_time:.3f}s",
        'concurrent_execution': True
    }

    # Save test results
    test_results_file = output_dir / "test_execution_results.json"
    with open(test_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results

def run_crew_with_output_capture(crew_folder, output_folder):
    """Run the crew and capture all output to files"""
    base_path, console_dir, results_dir, metadata_dir = setup_output_folders(output_folder)

    # Capture execution metadata
    execution_info = {
        "timestamp": datetime.now().isoformat(),
        "script": "run_crew.py",
        "crew_folder": str(crew_folder),
        "output_folder": str(base_path),
        "python_version": sys.version,
        "crewai_version": "1.9.3",  # You might want to get this dynamically
        "model": "bedrock/amazon.nova-micro-v1:0",
        "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    }

    # Files for output
    console_log = base_path / "crew_execution.log"
    final_result_file = base_path / "final_result.md"

    try:
        # Initialize Bedrock LLM
        print("🤖 Initializing AWS Bedrock Nova Micro...")
        llm = create_bedrock_llm()

        # Load agents from crew folder
        print(f"👥 Loading agents from {crew_folder}...")
        agents = load_agents_from_folder(crew_folder, llm)

        # Load tasks from crew folder
        print(f"📋 Loading tasks from {crew_folder}...")
        tasks = load_tasks_from_folder(crew_folder, agents)

        # Load and create crew
        print(f"⚡ Loading crew configuration from {crew_folder}...")
        crew = load_crew_from_folder(crew_folder, agents, tasks)

        # Execute the crew and capture the result
        result = crew.kickoff()

        # Save final result
        with open(final_result_file, 'w', encoding='utf-8') as f:
            f.write(str(result))

        # Extract and save code blocks as separate files
        extracted_files = extract_code_blocks(base_path, final_result_file)
        if extracted_files:
            execution_info["extracted_code_files"] = extracted_files

            # Execute extracted test functions and capture results
            test_results = run_extracted_tests(base_path)
            if test_results and test_results != {"error": "No test files found"}:
                execution_info["test_execution_results"] = test_results

                # Test results are now available for the test analyzer agent to review

        # Update execution info
        execution_info.update({
            "status": "success",
            "tasks_completed": len(tasks),
            "agents_used": [agent.role for agent in agents.values()]
        })

        print(f"\n{'='*50}")
        print("✅ Crew execution completed!")
        print(f"📁 Results saved to: {base_path}")
        print(f"📄 Final result saved to: {final_result_file}")
        if extracted_files:
            print(f"💾 Extracted {len(extracted_files)} function(s) as separate file(s):")
            for file in extracted_files:
                print(f"   ├── {Path(file).name}")

        return True

    except Exception as e:
        error_msg = f"❌ Error: {e}"
        print(error_msg)

        # Save error to file
        error_file = base_path / "error.log"
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
        print("5. Verify crew folder structure and YAML files")

        return False

    finally:
        # Save metadata regardless of success/failure
        save_execution_metadata(base_path, execution_info)

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
    # Generate timestamp for default output directory
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    default_output = f'.results/crew-{timestamp}'

    parser = argparse.ArgumentParser(
        description="Run CrewAI agents with AWS Bedrock Nova Micro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python run_crew.py --crew ./crews/example
  python run_crew.py --crew ./crews/example --output my_experiment_1
  python run_crew.py --crew ./crews/example --output ./.results/crew-{timestamp}
        """
    )
    parser.add_argument(
        '--crew', '-c',
        required=True,
        help='Path to crew configuration folder (required)'
    )
    parser.add_argument(
        '--output', '-o',
        default=default_output,
        help=f'Output folder for results (default: {default_output})'
    )

    args = parser.parse_args()

    # Validate crew folder exists
    crew_path = Path(args.crew)
    if not crew_path.exists():
        print(f"❌ Error: Crew folder '{args.crew}' does not exist")
        sys.exit(1)

    if not crew_path.is_dir():
        print(f"❌ Error: Crew path '{args.crew}' is not a directory")
        sys.exit(1)

    print("🚀 Starting CrewAI with AWS Bedrock Nova Micro")
    print("=" * 50)
    print(f"👥 Crew folder: {args.crew}")
    print(f"📁 Output folder: {args.output}")

    success = run_crew_with_output_capture(args.crew, args.output)

    if success:
        print(f"\n✅ All results saved to: {args.output}")
        print("📂 Files saved:")
        print(f"   ├── crew_execution.log")
        print(f"   ├── final_result.md")
        print(f"   └── execution_metadata.json")
    else:
        print(f"\n❌ Execution failed. Check error logs in: {args.output}/error.log")

if __name__ == "__main__":
    main()