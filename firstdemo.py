import os
from dotenv import load_dotenv


from openai import OpenAI

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
# Check the key

if not api_key:
    print("No API key was found - please head over to the troubleshooting notebook in this folder to identify & fix!")
elif not api_key.startswith("sk-proj-"):
    print("An API key was found, but it doesn't start sk-proj-; please check you're using the right key - see troubleshooting notebook")
elif api_key.strip() != api_key:
    print("An API key was found, but it looks like it might have space or tab characters at the start or end - please remove them - see troubleshooting notebook")
else:
    print("API key found and looks good so far!")
client = OpenAI()

message = "Hello, GPT! This is my first ever message to you! Hi!"

messages = [{"role": "user", "content": message}]

messages

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role":"user","content":"Hello"}]
)
print(response.choices[0].message.content)