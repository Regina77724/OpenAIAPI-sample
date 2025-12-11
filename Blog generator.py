import os
import json


from dotenv import load_dotenv
from openai import OpenAI
load_dotenv(override=True)
from openai import OpenAI
client = OpenAI()

# print(client.models.list())
user_topic = input("Enter your topic: ")
additional_info = input("Additional info: ")
custom_propmt = f"""you are a copy writer with years of experience writing impactful blog that
                    converge and help elevate brands. your task is to write a blog on any topic
                    system provides you with. make sure tp write in a format that works for medium.
                    each blog should be separated into segments that have titles and subtitles.
                    each paragraph should be three sentences long.
                    Topic : {user_topic}
                    Additional pointers : {additional_info}
                        """
#

response = client.completions.create( model= "gpt-3.5-turbo-instruct",
                           prompt= custom_propmt,
                           max_tokens= 700,
                                      temperature= 0.8)
print(response.choices[0].text)
