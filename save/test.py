import os
from openai import OpenAI
client = OpenAI(
api_key=os.environ.get("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")
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
        }
    ],
    model="gpt-4o"
)
print("Response:", chat_completion.choices[0].message.content)