import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI()

system_prompt = "You are a helpful customer support assistant."

while True:
    user_msg = input("You: ")
    if user_msg.lower() in ["exit", "quit"]:
        break

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
    )

    print("Bot:", response.output_text)
