﻿# Gemini DiscordBot using GeminiAPI
import os
import re
import aiohttp
import io
# import asyncio
import magic
import discord
from discord.ext import commands
from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_ID = "gemini-2.0-flash-exp"
# MODEL_ID = "gemini-exp-1206"
# MODEL_ID = "gemini-2.0-flash-thinking-exp" #Google API Alias
# MODEL_ID = "gemini-2.0-flash-thinking-exp-1219" #VertexAI
IMAGEN_MODEL='imagen-3.0-generate-001'


# Dictionary to store chat sessions
chat = {}

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# Google AI (API KEY)
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")

# VertexAI
# GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
# GCP_REGION = os.getenv("GCP_REGION")

# The maximum number of characters per Discord message
MAX_DISCORD_LENGTH = 2000

# Load the environment variable for enabling/disabling commands
IMG_COMMANDS_ENABLED = os.getenv('IMG_COMMANDS_ENABLED', 'False').lower() == 'true'
print(IMG_COMMANDS_ENABLED)

# Tool to support Google Search in Model
tools = [
    types.Tool(google_search=types.GoogleSearch())
]

generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    max_output_tokens = 8192,
    # safety_settings = [types.SafetySetting(
    #   category="HARM_CATEGORY_HATE_SPEECH",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_DANGEROUS_CONTENT",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
    #   threshold="OFF"
    # ),types.SafetySetting(
    #   category="HARM_CATEGORY_HARASSMENT",
    #   threshold="OFF"
    # )],
    tools = tools, #Comment out if you use gemini-2.0-flash-thinking-exp
  )

# Initialize Google AI via API_KEY
chat_model = genai.Client(api_key=GOOGLE_AI_KEY,  http_options={'api_version':'v1alpha'})

# Initialize Vertex AI API
# chat_model = genai.Client(
#     vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION
# )

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    """Triggered when the bot has successfully connected."""
    print("----------------------------------------")
    print(f'Gemini Bot Logged in as {bot.user}')
    print("----------------------------------------")

@bot.event
async def on_message(message):
    """Handle incoming messages."""
    if message.author == bot.user:
        return

    # Respond when mentioned or in DMs
    if message.mention_everyone:
        await message.channel.send(f'This is {bot.user}')
        return

    if message.content.startswith('!img'):
        await bot.process_commands(message)
        # For debug
        # print(message.content)
        # print(clean_discord_message(message.content))
        #
    elif bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        cleaned_text = clean_discord_message(message.content)
        async with message.channel.typing():
            if message.attachments:
                await process_attachments(message, cleaned_text)
            else:
                await process_text_message(message, cleaned_text)


def get_mime_type_from_bytes(byte_data):
    """Determine the MIME type of a given byte array."""
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(byte_data)
    # Replace unsupported MIME types starting with 'text/' with 'text/plain'
    if mime_type.startswith('text/'):
        mime_type = 'text/plain'
    return mime_type

async def process_attachments(message, cleaned_text):
    """Process message attachments and generate responses."""
    for attachment in message.attachments:
        await message.add_reaction('📄')
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(attachment.url) as resp:
                    if resp.status != 200:
                        await message.channel.send('Unable to download the file.')
                        return
                    file_data = await resp.read()
                    mime_type = get_mime_type_from_bytes(file_data)
                    response_text = await generate_response_with_file_and_text(message, file_data, cleaned_text, mime_type)
                    await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)
                    return
        except aiohttp.ClientError as e:
            await message.channel.send(f'Failed to download the file: {e}')
        except Exception as e:
            await message.channel.send(f'An unexpected error occurred: {e}')


async def process_text_message(message, cleaned_text):
    """Processes a text message and generates a response using a chat model."""
    if re.search(r'^RESET$', cleaned_text, re.IGNORECASE):
        chat.pop(message.author.id, None)
        await message.channel.send(f"🧹 History Reset for user: {message.author.name}")
        return

    await message.add_reaction('💬')
    response_text = await generate_response_with_text(message, cleaned_text)

    await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)

#################
from typing import Any
import logging

def process_answer(answer: Any) -> str:
    """
    Extract the text part from the LLM response for Discord.

    Args:
        answer: The LLM response object.

    Returns:
        The response string.
    """
    try:
        text = answer.text.strip()
        return text

    except AttributeError as e:
        logging.error(f"Error processing LLM response: Missing 'text' attribute. Answer object type: {type(answer).__name__}")
        return "Error: An issue occurred while processing the LLM response."

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        logging.exception(f"Unexpected error processing LLM response: type={error_type}, details={error_details}, Answer object type: {type(answer).__name__}")
        return "Error: An unexpected error has occurred. See system logs for more information."
    
##################
       
async def generate_response_with_text(message, cleaned_text):
    """Generate a response based on the provided text input."""
    global chat
    user_id = message.author.id
    chat_session = chat.get(user_id)
    if not chat_session:
        chat_session = chat_model.aio.chats.create(
            model=MODEL_ID,
            config=generate_content_config,
        )
        chat[user_id] = chat_session
    try:
        answer = await chat_session.send_message(cleaned_text)
        return process_answer(answer)
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the response."

async def generate_response_with_file_and_text(message, file, text, _mime_type):
    """Generate a response based on the provided file and text input."""
    global chat
    user_id = message.author.id
    chat_session = chat.get(user_id)
    if not chat_session:
        chat_session = chat_model.aio.chats.create(
            model=MODEL_ID,
            config=generate_content_config,
        )
        chat[user_id] = chat_session
    try:
        # file_like_object = io.BytesIO(file)
 
        text_part = f"\n{text if text else 'What is this?'}"
        file_byte = types.Part.from_bytes(data=file, mime_type=_mime_type)
        prompt_parts = [text_part, file_byte]


        answer = await chat_session.send_message(prompt_parts)
        return process_answer(answer)
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the response."

def clean_discord_message(input_string):
    """Remove special characters and Discord mentions from the message."""
    bracket_pattern = re.compile(r'<[^>]+>')
    return bracket_pattern.sub('', input_string)

async def split_and_send_messages(message_system, text, max_length):
    """Split the message into chunks and send them, respecting the maximum length."""
    start = 0
    while start < len(text):
        if len(text) - start <= max_length:
            await message_system.channel.send(text[start:])
            break

        end = start + max_length
        while end > start and text[end-1] not in ' \n\r\t':
            end -= 1

        if end == start:
            end = start + max_length

        await message_system.channel.send(text[start:end].strip())
        start = end

async def generate_image(prompt_text, negative_text, aspect_ratio):
    # ######################
    # Generate Image
    # ######################    
    response1 = chat_model.models.generate_image(
        model= IMAGEN_MODEL,
        prompt= prompt_text,
        config=types.GenerateImageConfig(
            negative_prompt= negative_text,
            number_of_images= 1,
            aspect_ratio = aspect_ratio,
            include_rai_reason= True,
            output_mime_type= "image/jpeg"
        )
    )

    image = response1.generated_images[0].image

    return image

async def handle_generation(ctx, prompt, negative_prompt, aspect_ratio):
    try:
        image = await generate_image(prompt, negative_prompt, aspect_ratio)
        with io.BytesIO(image.image_bytes) as image_binary:
            file = discord.File(image_binary, filename="generated_image.jpg")
            await ctx.send(file=file)

        file_data = image.image_bytes
        mime_type = get_mime_type_from_bytes(file_data)
        prompt_message = f"This image was generated by prompt: {prompt}; negative prompt: {negative_prompt}."
        response_text = await generate_response_with_file_and_text(ctx, file_data, prompt_message, mime_type)
        await split_and_send_messages(ctx, response_text, MAX_DISCORD_LENGTH)
    except ValueError as ve:
        await ctx.send(f"Invalid input provided for image generation: {str(ve)}")
    except Exception as e:
        await ctx.send(f"Failed to generate image: {str(e)}")

async def parse_args(args):
    parts = [part.strip() for part in args.split('|')]
    prompt = parts[0] if len(parts) > 0 else ""
    negative_prompt = ""
    aspect_ratio = "16:9"
    if len(parts) > 1:
        if re.match(r'^\d+:\d+$', parts[1]):
            aspect_ratio = parts[1]
        else:
            negative_prompt = parts[1]
    if len(parts) > 2:
        aspect_ratio = parts[2]
    return prompt, negative_prompt, aspect_ratio

@bot.command(name='img')
async def generate(ctx, *, args):
    if not IMG_COMMANDS_ENABLED:
        await ctx.send("The feature is currently disabled")
        return
    
    try:
        await ctx.message.add_reaction('🎨')
        prompt_text, negative_text, aspect_ratio = await parse_args(args)
        await ctx.send(f"Prompt: {prompt_text}\nNegative Prompt: {negative_text}")
        await handle_generation(ctx, prompt_text,negative_text, aspect_ratio)
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

#On Message Function
# Run the bot
bot.run(DISCORD_BOT_TOKEN)
