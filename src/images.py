from os import getenv

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

load_dotenv()

TOKEN = getenv("OPENAI_TOKEN")
MODEL = getenv("OPENAI_MODEL")

client = OpenAI(api_key=TOKEN)


def generate_image(prompt: str) -> tuple[str, bool]:
    try:
        response = client.images.generate(
            model=MODEL,
            prompt=prompt,
            response_format="url",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url, True
    except OpenAIError as error:
        return str(error), False
