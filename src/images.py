from os import getenv

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError

load_dotenv()

TOKEN = getenv("OPENAI_TOKEN")
MODEL = getenv("OPENAI_MODEL")

client = AsyncOpenAI(api_key=TOKEN)


async def generate_image(prompt: str) -> str | None:
    try:
        response = await client.images.generate(
            model=MODEL,
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except OpenAIError as error:
        return str(error)
