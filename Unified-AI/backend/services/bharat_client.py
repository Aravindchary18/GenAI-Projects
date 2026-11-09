import os
from openai import OpenAI


class BharatCodeClient:

    def __init__(self):

        api_key = os.getenv("BHARATCODE_API_KEY")

        if not api_key:
            raise ValueError("BHARATCODE_API_KEY is missing")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://bharatcode.ai/api/model/v1",
            timeout=120,
            max_retries=0
        )

        self.model = "qwen-3.8-27b"

    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        return response.choices[0].message.content

    def stream(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            stream=True
        )

        for chunk in response:

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                yield content