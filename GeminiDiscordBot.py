# Gemini DiscordBot using GeminiAPI
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
import datetime  # 追加: タイムスタンプ用

# MODEL_ID = "gemini-2.0-flash"
MODEL_ID = "gemini-2.0-pro-exp-02-05"
# MODEL_ID = "gemini-2.0-flash-thinking-exp" #Google API Alias
# MODEL_ID = "gemini-2.0-flash-thinking-exp-1219" #VertexAI
IMAGEN_MODEL='imagen-3.0-generate-002'


# Dictionary to store chat sessions
chat = {}

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# Google AI (API KEY)
# GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")

# VertexAI
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")

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
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    tools = tools,
    response_modalities=["TEXT"]
  )

# Initialize Google AI via API_KEY
# To use the thinking model you need to set your client to use the v1alpha version of the API:
# https://ai.google.dev/gemini-api/docs/grounding?lang=python
# chat_model = genai.Client(api_key=GOOGLE_AI_KEY,  http_options={'api_version':'v1alpha'})
# chat_model = genai.Client(api_key=GOOGLE_AI_KEY)

# Initialize Vertex AI API
chat_model = genai.Client(
    vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION
)

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
    elif bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        cleaned_text = clean_discord_message(message.content)
        
        # 追加: !save コマンドの検出
        save_to_file = False
        if cleaned_text.startswith("!save "):
            save_to_file = True
            cleaned_text = cleaned_text.replace("!save ", "", 1)
        
        async with message.channel.typing():
            if message.attachments:
                await process_attachments(message, cleaned_text, save_to_file)
            else:
                await process_text_message(message, cleaned_text, save_to_file)


def get_mime_type_from_bytes(byte_data):
    """Determine the MIME type of a given byte array."""
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(byte_data)
    # Replace unsupported MIME types starting with 'text/' with 'text/plain'
    if mime_type.startswith('text/'):
        mime_type = 'text/plain'
    return mime_type


async def process_attachments(message, cleaned_text, save_to_file=False):
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
                    
                    # 追加: !save コマンド処理
                    if save_to_file:
                        await save_response_as_file(message, response_text)
                    else:
                        await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)
                    return
        except aiohttp.ClientError as e:
            await message.channel.send(f'Failed to download the file: {e}')
        except Exception as e:
            await message.channel.send(f'An unexpected error occurred: {e}')



async def process_text_message(message, cleaned_text, save_to_file=False):
    """Processes a text message and generates a response using a chat model."""
    if re.search(r'^RESET$', cleaned_text, re.IGNORECASE):
        chat.pop(message.author.id, None)
        await message.channel.send(f"🧹 History Reset for user: {message.author.name}")
        return

    await message.add_reaction('💬')
    response_text = await generate_response_with_text(message, cleaned_text)
    
    # 追加: !save コマンド処理
    if save_to_file:
        await save_response_as_file(message, response_text)
    else:
        await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)


async def save_response_as_file(message, response_text):
    """
    Saves the response as a markdown file and sends it to the Discord channel.
    """
    # タイムスタンプ付きのファイル名を生成
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_response_{timestamp}.md"
    
    # 応答テキストを持つファイルオブジェクトを作成
    file = discord.File(io.StringIO(response_text), filename=filename)
    
    # ファイルを添付したメッセージを送信
    await message.channel.send(f"💾 Here's your response as a file:", file=file)
    
    # 最初の数行のプレビュー（オプション）
    preview_lines = response_text.split('\n')[:5]  # 最初の5行
    preview = '\n'.join(preview_lines)
    if len(preview_lines) >= 5:
        preview += "\n..."
    
    if preview.strip():  # 内容があればプレビューを送信
        await message.channel.send(f"📝 Preview:\n```\n{preview}\n```")

        
#################
from typing import Any
import logging

def process_answer(answer: Any) -> str:
    """
    Extracts the text part from the LLM response for Discord.
    This function first tries to get the text from answer.text. If that fails (e.g., due to a ValueError),
    it then checks answer.parts for any text. If no text is found there, it looks for text in
    answer.candidates[0].content.parts. If none of these contain meaningful text, it logs the entire
    answer object and returns an error message.

    Optionally, you can also log grounding metadata if available.

    Args:
        answer: The LLM response object.

    Returns:
        The extracted text if available, otherwise an error message.
    """
    try:
        # 1. Try to get text from answer.text
        try:
            text = answer.text.strip()
            return text
        except ValueError:
            # If answer.text raises a ValueError, move on to the next extraction method.
            pass

        # 2. Extract text from answer.parts if available.
        if hasattr(answer, 'parts') and answer.parts:
            text_parts = [part.text for part in answer.parts if hasattr(part, 'text') and part.text]
            if text_parts:
                return "\n".join(text_parts).strip()

        # 3. If candidates are present, extract text from the first candidate's content.parts.
        if hasattr(answer, 'candidates') and answer.candidates:
            candidate = answer.candidates[0]
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                candidate_text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
                if candidate_text_parts:
                    return "\n".join(candidate_text_parts).strip()
            
            # Optionally, log the grounding metadata if available.
            if hasattr(candidate, 'groundingMetadata') and candidate.groundingMetadata:
                logging.info("Grounding metadata (rendered content): %s", candidate.groundingMetadata.searchEntryPoint.renderedContent)

        # If no meaningful text is found, log the entire answer object.
        logging.error("No text found in Gemini response. Answer content: %s", answer)
        return "Error: No text found in Gemini response."

    except AttributeError as e:
        logging.error("Error processing LLM response: Missing attribute. Answer object type: %s, Answer content: %s",
                    type(answer).__name__, answer)
        return "Error: An issue occurred while processing the LLM response."

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        logging.exception("Unexpected error processing LLM response: type=%s, details=%s, Answer object type: %s, Answer content: %s",
                        error_type, error_details, type(answer).__name__, answer)
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


import logging

async def generate_image(prompt_text, aspect_ratio):
    try:
        response1 = chat_model.models.generate_images(
            model=IMAGEN_MODEL,
            prompt=prompt_text,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                include_rai_reason=True,
                output_mime_type="image/jpeg",
                person_generation="ALLOW_ALL",
                safety_filter_level="BLOCK_ONLY_HIGH",
            )
        )

        # Log the full response for debugging purposes
        logging.info("Full response: %s", response1)

        if not response1 or not hasattr(response1, 'generated_images') or not response1.generated_images:
            logging.error("Failed to generate image. API response is invalid. Response: %s", response1)
            raise ValueError("Failed to generate image. API response is invalid.")

        # Loop through the generated images (in case multiple images are generated)
        for idx, generated_img in enumerate(response1.generated_images):
            # If rai_filtered_reason is present, log it
            if hasattr(generated_img, 'rai_filtered_reason') and generated_img.rai_filtered_reason:
                logging.warning(f"[Index {idx}] RAI Filtered reason: {generated_img.rai_filtered_reason}")

            # Check if the image object or image_bytes is None
            if (
                generated_img.image is None
                or not hasattr(generated_img.image, 'image_bytes')
                or generated_img.image.image_bytes is None
            ):
                # If rai_filtered_reason is available, use it as the error message
                if hasattr(generated_img, 'rai_filtered_reason') and generated_img.rai_filtered_reason:
                    error_msg = f"Failed to generate image. {generated_img.rai_filtered_reason}"
                else:
                    error_msg = "Failed to generate image. 'image_bytes' is missing or None."
                logging.error(error_msg)
                raise ValueError(error_msg)

        # Here, we return only one image as an example
        return response1.generated_images[0].image

    except Exception as e:
        logging.error("Error in generate_image: %s", e, exc_info=True)
        raise


async def handle_generation(ctx, prompt, aspect_ratio):
    try:
        image = await generate_image(prompt, aspect_ratio)
        with io.BytesIO(image.image_bytes) as image_binary:
            file = discord.File(image_binary, filename="generated_image.jpg")
            await ctx.send(file=file)

        file_data = image.image_bytes
        mime_type = get_mime_type_from_bytes(file_data)
        prompt_message = f"This image was generated by prompt: {prompt}."
        response_text = await generate_response_with_file_and_text(ctx, file_data, prompt_message, mime_type)  # Added mime_type
        await split_and_send_messages(ctx, response_text, MAX_DISCORD_LENGTH)
    except ValueError as ve:  # Catch ValueError specifically
        await ctx.send(f"Invalid input or API error: {str(ve)}")
    except Exception as e:
        await ctx.send(f"Failed to generate image: {str(e)}")

async def parse_args(args):
    parts = [part.strip() for part in args.split('|')]
    prompt = parts[0] if len(parts) > 0 else ""
    aspect_ratio = "16:9"
    if len(parts) > 1:
        aspect_ratio = parts[1]
    return prompt, aspect_ratio

@bot.command(name='img')
async def generate(ctx, *, args):
    if not IMG_COMMANDS_ENABLED:
        await ctx.send("The feature is currently disabled")
        return
    
    try:
        await ctx.message.add_reaction('🎨')
        prompt_text, aspect_ratio = await parse_args(args)
        await ctx.send(f"Prompt: {prompt_text}")
        await handle_generation(ctx, prompt_text, aspect_ratio)
    except Exception as e:
        await ctx.send(f"An error occurred: {str(e)}")

#On Message Function
# Run the bot
bot.run(DISCORD_BOT_TOKEN)
