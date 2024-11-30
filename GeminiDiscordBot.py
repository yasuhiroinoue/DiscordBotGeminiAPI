# Gemini DiscordBot using vertexai
import os
import re
import aiohttp
import io
import asyncio
import magic
import discord
from discord.ext import commands
from dotenv import load_dotenv
import google.generativeai as genai
from io import BytesIO

# Dictionary to store chat sessions
chat = {}

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")

# The maximum number of characters per Discord message
MAX_DISCORD_LENGTH = 2000

# Configure the generative AI model
text_generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "max_output_tokens": 8192,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"}
]

# Initialize generative AI model
genai.configure(api_key=GOOGLE_AI_KEY)
chat_model = genai.GenerativeModel(model_name="gemini-exp-1121", generation_config=text_generation_config, safety_settings=safety_settings)

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

    if message.content.startswith('!img') or message.content.startswith('!fimg'):
        await bot.process_commands(message)
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

async def async_send_message(chat_session, prompt): 
    """Send a message asynchronously using a chat session."""
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(None, chat_session.send_message, prompt)
        return response
    except Exception as e:
        print(f"Error sending message: {e}")
        return None

async def process_text_message(message, cleaned_text):
    """Processes a text message and generates a response using a chat model."""
    if re.search(r'^RESET$', cleaned_text, re.IGNORECASE):
        chat.pop(message.author.id, None)
        await message.channel.send(f"🧹 History Reset for user: {message.author.name}")
        return

    await message.add_reaction('💬')
    response_text = await generate_response_with_text(message, cleaned_text)
    await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)

async def generate_response_with_text(message, cleaned_text):
    """Generate a response based on the provided text input."""
    global chat
    user_id = message.author.id
    chat_session = chat.get(user_id)
    if not chat_session:
        chat_session = chat_model.start_chat()
        chat[user_id] = chat_session
    try:
        answer = await async_send_message(chat_session, cleaned_text)
        if answer.candidates and answer.candidates[0].content.parts:
            return answer.candidates[0].content.parts[0].text
        else:
            return "No valid response received."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the response."

async def generate_response_with_file_and_text(message, file, text, _mime_type):
    """Generate a response based on the provided file and text input."""
    file_like_object = BytesIO(file)
    file_part = genai.upload_file(file_like_object, mime_type=_mime_type)
    text_part = f"\n{text if text else 'What is this?'}"
    prompt_parts = [file_part, text_part]

    global chat
    user_id = message.author.id
    chat_session = chat.get(user_id)
    if not chat_session:
        chat_session = chat_model.start_chat()
        chat[user_id] = chat_session
    try:
        answer = await async_send_message(chat_session, prompt_parts)
        if answer.candidates and answer.candidates[0].content.parts:
            return answer.candidates[0].content.parts[0].text
        else:
            return "No valid response received."
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

# Run the bot
bot.run(DISCORD_BOT_TOKEN)
