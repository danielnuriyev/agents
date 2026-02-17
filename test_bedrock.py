#!/usr/bin/env python3
"""
Simple AWS Bedrock Test with Amazon Nova Micro

This script sends one simple question to AWS Bedrock Nova Micro.

Prerequisites:
- AWS credentials configured
- Required packages: pip install boto3 python-dotenv

Usage:
    python test_bedrock.py
"""

import os
import boto3
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def ask_bedrock(question):
    """Send a simple question to Amazon Nova Micro"""
    try:
        # Create Bedrock client
        client = boto3.client(
            'bedrock-runtime',
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )

        # Prepare the request
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": question}]
                }
            ],
            "inferenceConfig": {
                "temperature": 0.7,
                "maxTokens": 100
            }
        }

        # Invoke the model
        response = client.invoke_model(
            modelId="amazon.nova-micro-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        # Parse and return the response
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']

    except Exception as e:
        return f"Error: {e}"

def main():
    """Send one simple question to Bedrock"""
    print("🤖 Testing AWS Bedrock Nova Micro")
    print("-" * 30)

    # Ask one simple question
    question = "What is 2 + 2?"
    print(f"❓ Question: {question}")

    answer = ask_bedrock(question)
    print(f"✅ Answer: {answer}")

if __name__ == "__main__":
    main()