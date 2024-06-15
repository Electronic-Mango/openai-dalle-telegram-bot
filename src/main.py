from logging import INFO, basicConfig
from os import getenv

from dotenv import load_dotenv
from loguru import logger
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler
from telegram.ext.filters import COMMAND, TEXT

from images import generate_image
from user_filer import USER_FILTER


def main() -> None:
    load_dotenv()
    basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=INFO)
    bot = ApplicationBuilder().token(getenv("BOT_TOKEN")).build()
    bot.add_handler(MessageHandler(USER_FILTER & TEXT & ~COMMAND, image_response))
    bot.add_handler(CommandHandler({"again"}, regenerate_image, USER_FILTER))
    bot.add_handler(CommandHandler({"help", "start"}, start, USER_FILTER))
    bot.run_polling()


async def image_response(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = update.message.text
    context.chat_data["prompt"] = prompt
    await _generate_image(update, prompt)


async def regenerate_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prompt = context.chat_data.get("prompt")
    if prompt:
        await _generate_image(update, prompt)
    else:
        await update.message.reply_text("No prompt found!")


async def _generate_image(update: Update, prompt: str) -> None:
    chat_id = update.message.chat_id
    logger.info(f"[{chat_id}] Generating image for prompt [{prompt}]")
    await update.message.reply_text("Generating...")
    image_url = await generate_image(prompt)
    logger.info(f"[{chat_id}] Generated [{image_url}]")
    await update.message.reply_photo(image_url)


async def start(update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Send me a prompt and I'll respond with an image!")


if __name__ == "__main__":
    main()
