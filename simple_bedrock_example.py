#!/usr/bin/env python3
"""
Simple AWS Bedrock Example with Amazon Nova Micro

This script demonstrates basic AWS Bedrock functionality with Amazon Nova Micro
without requiring the full CrewAI framework.

Prerequisites:
- AWS credentials configured
- Required packages: pip install langchain-aws boto3 python-dotenv

Usage:
    python simple_bedrock_example.py
"""

import os
import boto3
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def invoke_nova_micro_direct(prompt):
    """Invoke Amazon Nova Micro using boto3 directly"""
    try:
        # Create Bedrock client
        client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )

        # Prepare the request for Nova Micro
        # Nova Micro expects messages format with content as array
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "temperature": 0.7,
                "maxTokens": 1000
            }
        }

        # Convert to JSON string
        body_json = json.dumps(body)

        # Invoke the model
        response = client.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=body_json,
            contentType="application/json",
            accept="application/json"
        )

        # Parse the response
        response_body = json.loads(response['body'].read())

        # Extract the generated text
        if 'output' in response_body and 'message' in response_body['output']:
            return response_body['output']['message']['content'][0]['text']
        else:
            return f"Unexpected response format: {response_body}"

    except Exception as e:
        return f"Error: {e}"

def coding_assistant_task():
    """Simulate a coding assistant task"""
    print("🤖 Coding Assistant Task")
    print("-" * 40)

    prompt = """
    Create a Python function that takes a list of dictionaries containing user data
    and returns the average age. Each dictionary has 'name' and 'age' keys.

    Example input: [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
    Example output: 27.5

    Include error handling for invalid data.
    """

    response = invoke_nova_micro_direct(prompt)
    print("📝 Generated Code:")
    print(response)
    return response

def code_review_task(code):
    """Simulate a code review task"""
    if not code or code.startswith("Error:"):
        print("⚠️  No code to review")
        return

    print("\n🔍 Code Review Task")
    print("-" * 40)

    prompt = f"""
    Review the following Python code for:
    1. Code quality and readability
    2. Error handling
    3. Best practices
    4. Potential improvements

    Code to review:
    {code}

    Provide specific recommendations.
    """

    response = invoke_nova_micro_direct(prompt)
    print("📋 Review Feedback:")
    print(response)
    return response

def main():
    """Main execution function"""
    print("🚀 Simple AWS Bedrock Nova Micro Demo")
    print("=" * 50)

    try:
        # Test basic connectivity
        print("🧪 Testing basic connectivity...")
        test_response = invoke_nova_micro_direct("Say hello in exactly 3 words.")
        print(f"✅ Connection successful: {test_response}")

        # Run coding assistant task
        generated_code = coding_assistant_task()

        # Run code review task
        code_review_task(generated_code)

        print("\n" + "=" * 50)
        print("✅ Demo completed successfully!")
        print("🎉 AWS Bedrock Nova Micro is working!")

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check AWS credentials: aws configure")
        print("2. Verify AWS region has Bedrock access")
        print("3. Ensure Nova Micro model is available")
        print("4. Check IAM permissions for bedrock:InvokeModel")

if __name__ == "__main__":
    main()