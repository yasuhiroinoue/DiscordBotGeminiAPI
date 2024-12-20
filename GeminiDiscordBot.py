﻿# Gemini DiscordBot using GeminiAPI
import os
import re
import aiohttp
import io
import asyncio
import magic
import discord
from discord.ext import commands
from dotenv import load_dotenv
from google import genai
from google.genai import types

MODEL_ID = "gemini-2.0-flash-exp"
IMAGEN_MODEL='imagen-3.0-generate-001'


# MODEL_ID = "gemini-exp-1206"

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
    response_modalities = ["TEXT"],
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
  )

# Initialize Google AI via API_KEY
# chat_model = genai.Client(api_key=GOOGLE_AI_KEY)

# Only run this block for Vertex AI API
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

async def async_send_message(chat_session, prompt): 
    """Send a message asynchronously using a chat session."""
    loop = asyncio.get_running_loop()
    try:
        response = await loop.run_in_executor(None, chat_session.send_message, prompt)
        # For debug
        # print(response)
        # print("Response parts:")
        # for part in response.candidates[0].content.parts:
        #     print(part.text)
        #
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

def process_answer(answer):
    """
    Process the LLM response to concatenate all parts and handle grounding metadata.
    """
    if answer.candidates and answer.candidates[0].content.parts:
        # 全ての parts を連結して取得
        original_text = "".join([part.text for part in answer.candidates[0].content.parts if part.text])

        candidate = answer.candidates[0]
        gm = candidate.grounding_metadata

        if not gm.grounding_supports:
            # grounding_supports が空の場合は参照をつけずにそのまま出力
            return original_text
        else:
            # grounding_supports がある場合は参照を処理
            import re
            ref_dict = {}

            for support in gm.grounding_supports:
                text = support.segment.text
                refs_found = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text)
                # 引用番号が見つからない場合は次の support へ
                if not refs_found:
                    continue
                
                for ref_group in refs_found:
                    ref_nums = [r.strip() for r in ref_group.split(',')]
                    # 引用番号の数と grounding_chunk_indices が一致する場合のみ処理
                    if len(ref_nums) == len(support.grounding_chunk_indices):
                        for i, ref_num in enumerate(ref_nums):
                            idx = support.grounding_chunk_indices[i]
                            # index 範囲チェック
                            if idx < len(gm.grounding_chunks):
                                chunk = gm.grounding_chunks[idx]
                                if chunk.web:
                                    uri = chunk.web.uri
                                    if ref_num not in ref_dict:
                                        ref_dict[ref_num] = uri

            if ref_dict:
                ref = "References:\n"
                for ref_num in sorted(ref_dict, key=lambda x: int(x)):
                    ref += f"[{ref_num}] {ref_dict[ref_num]}\n"
                return f"{original_text}\n{ref}"
            else:
                # 引用があるはずなのに ref_dict が空の場合、または refs が無かった場合
                return original_text
    else:
        return "No valid response received."

async def generate_response_with_text(message, cleaned_text):
    """Generate a response based on the provided text input."""
    global chat
    user_id = message.author.id
    chat_session = chat.get(user_id)
    if not chat_session:
        chat_session = chat_model.chats.create(
            model=MODEL_ID,
            config=generate_content_config,
        )
        chat[user_id] = chat_session
    try:
        answer = await async_send_message(chat_session, cleaned_text)
        return process_answer(answer)

        # if answer.candidates and answer.candidates[0].content.parts:
            
        #     # 全ての parts を連結して取得
        #     original_text = "".join([part.text for part in answer.candidates[0].content.parts if part.text])
        #     # return response_text
        #     # return answer.candidates[0].content.parts[0].text
        #     candidate = answer.candidates[0]
        #     gm = candidate.grounding_metadata
        #     # original_text = candidate.content.parts[0].text

        #     if not gm.grounding_supports:
        #         # grounding_supportsが空の場合は参照をつけずにそのまま出力
        #         response_text = original_text
        #         return response_text
        #     else:
        #         # print("grounding_supportsがある場合は、ここに先ほどの処理を入れる")
        #         import re
                
        #         ref_dict = {}
                
        #         for support in gm.grounding_supports:
        #             import re

        #             candidate = answer.candidates[0]
        #             gm = candidate.grounding_metadata

        #             if not gm.grounding_supports:
        #                 # grounding_supportsが空の場合は参照をつけずにそのまま出力
        #                 response_text = original_text
        #             else:
        #                 ref_dict = {}
        #                 for support in gm.grounding_supports:
        #                     text = support.segment.text
        #                     refs_found = re.findall(r'\[(\d+(?:,\s*\d+)*)\]', text)
        #                     # 引用番号が見つからない場合は次のsupportへ
        #                     if not refs_found:
        #                         continue
                            
        #                     for ref_group in refs_found:
        #                         ref_nums = [r.strip() for r in ref_group.split(',')]
        #                         # 引用番号の数とgrounding_chunk_indicesが一致する場合のみ処理
        #                         if len(ref_nums) == len(support.grounding_chunk_indices):
        #                             for i, ref_num in enumerate(ref_nums):
        #                                 idx = support.grounding_chunk_indices[i]
        #                                 # index範囲チェック
        #                                 if idx < len(gm.grounding_chunks):
        #                                     chunk = gm.grounding_chunks[idx]
        #                                     if chunk.web:
        #                                         uri = chunk.web.uri
        #                                         if ref_num not in ref_dict:
        #                                             ref_dict[ref_num] = uri

        #                 if ref_dict:
        #                     ref = "References:\n"
        #                     for ref_num in sorted(ref_dict, key=lambda x: int(x)):
        #                         ref += f"[{ref_num}] {ref_dict[ref_num]}\n"
        #                     response_text = f"{original_text}\n{ref}"
        #                 else:
        #                     # 引用があるはずなのにref_dictが空の場合、またはrefsが無かった場合
        #                     response_text = original_text

        #             return response_text

        # else:
        #     return "No valid response received."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "An error occurred while generating the response."

async def generate_response_with_file_and_text(message, file, text, _mime_type):
    """Generate a response based on the provided file and text input."""
    global chat
    user_id = message.author.id
    chat_session = chat.get(user_id)
    if not chat_session:
        chat_session = chat_model.chats.create(
            model=MODEL_ID,
            config=generate_content_config,
        )
        chat[user_id] = chat_session
    try:
        # file_like_object = io.BytesIO(file)
 
        text_part = f"\n{text if text else 'What is this?'}"
        file_byte = types.Part.from_bytes(data=file, mime_type=_mime_type)
        prompt_parts = [text_part, file_byte]


        answer = await async_send_message(chat_session, prompt_parts)
        return process_answer(answer)
        # if answer.candidates and answer.candidates[0].content.parts:
        #     return answer.candidates[0].content.parts[0].text
        # else:
        #     return "No valid response received."
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
