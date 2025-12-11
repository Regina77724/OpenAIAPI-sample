import os
import json
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)

from openai import OpenAI
client = OpenAI()

def agent(role, user_input):
    return client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": role},
            {"role": "user", "content": user_input}
        ]
    ).output_text

while True:
    task = input("Task: ")

    planner = agent("You are a planning agent.", task)
    researcher = agent("You are a research agent.", planner)
    writer = agent("You are a writing agent.", researcher)

    print("\nPlan:", planner)
    print("Research:", researcher)
    print("Final Output:", writer)
