# Gemini DiscordBot using GeminiAPI
import os
import re
import aiohttp
import io

import asyncio
import magic
import discord
import tempfile
import urllib.parse
from discord.ext import commands
from dotenv import load_dotenv
from google import genai
from google.genai import types
import datetime  # Added: For timestamp
import logging  # Added: logging module

# Dictionary to store chat sessions
chat = {}
# Dictionary to store image generation chat sessions
image_chat = {}

# Load environment variables
load_dotenv()

MODEL_ID = os.getenv("MODEL_ID")
# IMAGEN_MODEL = os.getenv("IMAGEN_MODEL") # Deprecated
GEMINI_IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")

DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
# Google AI (API KEY)
# GOOGLE_AI_KEY = os.getenv("GOOGLE_AI_KEY")

# VertexAI
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_REGION = os.getenv("GCP_REGION")

# The maximum number of characters per Discord message
MAX_DISCORD_LENGTH = 2000

# Supported aspect ratios for Gemini image generation
SUPPORTED_ASPECT_RATIOS = frozenset({"1:1", "16:9", "9:16", "4:3", "3:4"})
DEFAULT_ASPECT_RATIO = "1:1"

# Load the environment variable for enabling/disabling commands
IMG_COMMANDS_ENABLED = os.getenv("IMG_COMMANDS_ENABLED", "False").lower() == "true"
print(f"Image commands enabled: {IMG_COMMANDS_ENABLED}")

# Debug settings
DEBUG_SAVE_CLOUD_FILES = os.getenv("DEBUG_SAVE_CLOUD_FILES", "False").lower() == "true"
DEBUG_FILES_DIR = os.getenv("DEBUG_FILES_DIR", "debug_files")
# Setting for logging user IDs (for debugging)
DEBUG_LOG_USER_IDS = os.getenv("DEBUG_LOG_USER_IDS", "False").lower() == "true"

# Create debug directory (if necessary)
if DEBUG_SAVE_CLOUD_FILES and not os.path.exists(DEBUG_FILES_DIR):
    os.makedirs(DEBUG_FILES_DIR)
    print(f"Created debug files directory: {DEBUG_FILES_DIR}")

if DEBUG_SAVE_CLOUD_FILES:
    print("Debug mode: Cloud files will be saved locally")

if DEBUG_LOG_USER_IDS:
    print("Debug mode: User ID logging is enabled")
else:
    print("Production mode: User ID logging is disabled")

# Load allowed user IDs from environment variable (comma-separated)
ALLOWED_USER_IDS_STR = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = set(
    int(uid.strip()) for uid in ALLOWED_USER_IDS_STR.split(",") if uid.strip().isdigit()
)
if not ALLOWED_USER_IDS:
    print("Warning: No ALLOWED_USER_IDS set. Bot will respond to everyone.")
else:
    print(f"Allowed user IDs: {ALLOWED_USER_IDS}")

# Configure logging for user interactions
log_formatter = logging.Formatter(
    "%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
log_handler = logging.FileHandler("bot_usage.log", encoding="utf-8")  # Log file name
log_handler.setFormatter(log_formatter)

usage_logger = logging.getLogger("bot_usage")
usage_logger.setLevel(logging.INFO)
usage_logger.addHandler(log_handler)
# Avoid propagating to root logger to prevent duplicate console logs if root is configured
usage_logger.propagate = False

# Update notification messages
if DEBUG_LOG_USER_IDS:
    print("Logging user interactions with IDs to bot_usage.log")
else:
    print("Logging user interactions without user IDs to bot_usage.log")

# Tool to support Google Search in Model
tools = [types.Tool(google_search=types.GoogleSearch())]

generate_content_config = types.GenerateContentConfig(
    temperature=1,
    top_p=0.95,
    max_output_tokens=65535,
    safety_settings=[
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"
        ),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
    ],
    tools=tools,
    response_modalities=["TEXT"],
)

# Initialize Google AI via API_KEY
# To use the thinking model you need to set your client to use the v1alpha version of the API:
# https://ai.google.dev/gemini-api/docs/grounding?lang=python
# chat_model = genai.Client(api_key=GOOGLE_AI_KEY,  http_options={'api_version':'v1alpha'})
# chat_model = genai.Client(api_key=GOOGLE_AI_KEY)

# Initialize Vertex AI API
chat_model = genai.Client(vertexai=True, project=GCP_PROJECT_ID, location=GCP_REGION)

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True


class GeminiBot(commands.Bot):
    async def setup_hook(self) -> None:
        self.http_session = aiohttp.ClientSession()

    async def close(self) -> None:
        if getattr(self, "http_session", None) is not None:
            await self.http_session.close()
        await super().close()


bot = GeminiBot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    """Triggered when the bot has successfully connected."""
    print("----------------------------------------")
    print(f"Gemini Bot Logged in as {bot.user}")
    print("----------------------------------------")


@bot.event
async def on_message(message):
    """Handle incoming messages."""
    if message.author == bot.user:
        return

    # Check if the author is in the allowed list (if the list is defined)
    if ALLOWED_USER_IDS and message.author.id not in ALLOWED_USER_IDS:
        print(
            f"Ignoring message from unauthorized user: {message.author.name} ({message.author.id})"
        )
        return  # Ignore messages from non-allowed users

    # Log the interaction only for DMs and only if debugging is enabled
    if isinstance(message.channel, discord.DMChannel) and DEBUG_LOG_USER_IDS:
        channel_info = f"DM ({message.channel.id})"
        usage_logger.info(
            f"Processing message from User: {message.author.name} (ID: {message.author.id}) in {channel_info}"
        )

    # Respond when mentioned or in DMs
    if message.mention_everyone:
        await message.channel.send(f"This is {bot.user}")
        return

    # bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        # Extract cloud storage links from the message
        original_content = message.content
        cleaned_text, cloud_links = extract_cloud_links(
            clean_discord_message(original_content)
        )

        # Command detection
        save_to_file = False
        img_command = False
        edit_command = False

        # Detect !save command
        if cleaned_text.startswith("!save "):
            save_to_file = True
            cleaned_text = cleaned_text.replace("!save ", "", 1)

        # Detect !img command
        elif cleaned_text.startswith("!img "):
            if not IMG_COMMANDS_ENABLED:
                await message.channel.send(
                    "The image generation feature is currently disabled"
                )
                return
            img_command = True
            prompt_text = cleaned_text.replace("!img ", "", 1)

        # Detect !edit command
        elif cleaned_text.startswith("!edit "):
            if not IMG_COMMANDS_ENABLED:
                await message.channel.send(
                    "The image generation feature is currently disabled"
                )
                return
            edit_command = True
            prompt_text = cleaned_text.replace("!edit ", "", 1)

        async with message.channel.typing():
            # Process cloud storage link (if exists)
            if cloud_links:
                try:
                    # Process only the first link (if multiple exist)
                    file_data, mime_type = await download_from_cloud_storage(
                        message, cloud_links[0]
                    )

                    await process_cloud_file(
                        message, cleaned_text, file_data, mime_type, save_to_file
                    )
                except Exception as e:
                    await message.channel.send(
                        f"Failed to process cloud storage link: {str(e)}"
                    )
                    # Do not continue normal flow if there is an error
                    return

            # Normal processing flow
            elif img_command:
                await message.add_reaction("🎨")
                # Process !img command
                prompt_text, aspect_ratio, invalid_ratio = await parse_args(prompt_text)
                if invalid_ratio is not None:
                    supported = ", ".join(sorted(SUPPORTED_ASPECT_RATIOS))
                    await message.channel.send(
                        f"Unsupported aspect_ratio `{invalid_ratio}`. "
                        f"Supported values: {supported}. Falling back to `{DEFAULT_ASPECT_RATIO}`."
                    )
                await message.channel.send(f"Prompt: {prompt_text}")
                await handle_generation(message, prompt_text, aspect_ratio)

            elif edit_command:
                await message.add_reaction("✏️")
                # Process !edit command
                await message.channel.send(f"Editing: {prompt_text}")
                await handle_edit_generation(message, prompt_text)

            elif message.attachments:
                await process_attachments(message, cleaned_text, save_to_file)
            else:
                await process_text_message(message, cleaned_text, save_to_file)


def get_mime_type_from_bytes(byte_data):
    """Determine the MIME type of a given byte array."""
    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(byte_data)
    # Replace unsupported MIME types starting with 'text/' with 'text/plain'
    if mime_type.startswith("text/"):
        mime_type = "text/plain"
    return mime_type


async def process_attachments(message, cleaned_text, save_to_file=False):
    """Process message attachments and generate responses."""
    files_data = []

    # Collect all attachments
    for attachment in message.attachments:
        try:
            async with bot.http_session.get(attachment.url) as resp:
                if resp.status != 200:
                    await message.channel.send(f"Unable to download the file: {attachment.filename}")
                    continue
                file_data = await resp.read()
                mime_type = get_mime_type_from_bytes(file_data)
                files_data.append({"data": file_data, "mime_type": mime_type})
        except aiohttp.ClientError as e:
            logging.exception("Failed to download attachment %s", attachment.filename)
            await message.channel.send(f"Failed to download the file {attachment.filename}: {e}")
        except Exception as e:
            logging.exception("Unexpected error while downloading attachment %s", attachment.filename)
            await message.channel.send(f"An unexpected error occurred with {attachment.filename}: {e}")

    if not files_data:
        return

    await message.add_reaction("📄")
    
    try:
        response_text = await generate_response_with_files_and_text(
            message, files_data, cleaned_text
        )

        # Added: !save command processing
        if save_to_file:
            await save_response_as_file(message, response_text)
        else:
            await split_and_send_messages(
                message, response_text, MAX_DISCORD_LENGTH
            )
    except Exception as e:
        logging.exception("Unexpected error during response generation for attachments")
        await message.channel.send(f"An unexpected error occurred during generation: {e}")


async def process_text_message(message, cleaned_text, save_to_file=False):
    """Processes a text message and generates a response using a chat model."""
    if re.search(r"^RESET$", cleaned_text, re.IGNORECASE):
        chat.pop(message.author.id, None)
        image_chat.pop(message.author.id, None)
        await message.channel.send(f"🧹 History (Text & Image) Reset for user: {message.author.name}")
        return

    await message.add_reaction("💬")
    response_text = await generate_response_with_text(message, cleaned_text)

    # Added: !save command processing
    if save_to_file:
        await save_response_as_file(message, response_text)
    else:
        await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)


async def save_response_as_file(message, response_text):
    """
    Saves the response as a markdown file and sends it to the Discord channel.
    """
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_response_{timestamp}.md"

    # Create file object with response text
    file = discord.File(io.StringIO(response_text), filename=filename)

    # Send message with attached file
    await message.channel.send(f"💾 Here's your response as a file:", file=file)

    # Preview of the first few lines (optional)
    preview_lines = response_text.split("\n")[:5]  # First 5 lines
    preview = "\n".join(preview_lines)
    if len(preview_lines) >= 5:
        preview += "\n..."

    if preview.strip():  # Send preview if content exists
        await message.channel.send(f"📝 Preview:\n```\n{preview}\n```")


#################
from typing import Any


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
        if hasattr(answer, "parts") and answer.parts:
            text_parts = [
                part.text
                for part in answer.parts
                if hasattr(part, "text") and part.text
            ]
            if text_parts:
                return "\n".join(text_parts).strip()

        # 3. If candidates are present, extract text from the first candidate's content.parts.
        if hasattr(answer, "candidates") and answer.candidates:
            candidate = answer.candidates[0]
            if (
                hasattr(candidate, "content")
                and candidate.content
                and hasattr(candidate.content, "parts")
            ):
                candidate_text_parts = [
                    part.text
                    for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                ]
                if candidate_text_parts:
                    return "\n".join(candidate_text_parts).strip()

            # Optionally, log the grounding metadata if available.
            if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                logging.info(
                    "Grounding metadata (rendered content): %s",
                    candidate.grounding_metadata.search_entry_point.rendered_content,
                )

        # If no meaningful text is found, log the entire answer object.
        logging.error("No text found in Gemini response. Answer content: %s", answer)
        return "Error: No text found in Gemini response."

    except AttributeError as e:
        logging.error(
            "Error processing LLM response: Missing attribute. Answer object type: %s, Answer content: %s",
            type(answer).__name__,
            answer,
        )
        return "Error: An issue occurred while processing the LLM response."

    except Exception as e:
        error_type = type(e).__name__
        error_details = str(e)
        logging.exception(
            "Unexpected error processing LLM response: type=%s, details=%s, Answer object type: %s, Answer content: %s",
            error_type,
            error_details,
            type(answer).__name__,
            answer,
        )
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
    except Exception:
        logging.exception("Error in generate_response_with_text for user %s", user_id)
        return "An error occurred while generating the response."


async def generate_response_with_files_and_text(message, files, text):
    """Generate a response based on the provided files and text input.
    
    Args:
        message: Discord message object.
        files (list): List of dicts, each containing 'data' (bytes) and 'mime_type' (str).
        text (str): The text prompt.
    """
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
        prompt_parts = [text if text else 'What is this?']
        
        for file_info in files:
            file_byte = types.Part.from_bytes(data=file_info['data'], mime_type=file_info['mime_type'])
            prompt_parts.append(file_byte)

        answer = await chat_session.send_message(prompt_parts)
        return process_answer(answer)

    except Exception:
        logging.exception("Error in generate_response_with_files_and_text for user %s", user_id)
        return "An error occurred while generating the response."


def clean_discord_message(input_string):
    """Remove special characters and Discord mentions from the message."""
    bracket_pattern = re.compile(r"<[^>]+>")
    return bracket_pattern.sub("", input_string)


def save_debug_file(file_data, service_name, original_url, mime_type):
    """Save the downloaded file locally for debugging.

    Args:
        file_data (bytes): File data
        service_name (str): Service name (e.g., "dropbox", "gdrive", "onedrive")
        original_url (str): Original URL (used for filename)
        mime_type (str): MIME type of the file

    Returns:
        str or None: Path of the saved file, or None if saving failed
    """
    if not DEBUG_SAVE_CLOUD_FILES:
        return None

    # Attempt to infer filename from URL
    url_filename = os.path.basename(urllib.parse.urlparse(original_url).path)
    if not url_filename or url_filename == "":
        # If filename cannot be obtained from URL, use only timestamp
        url_filename = ""

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Infer extension from MIME type
    extension = ""
    if "/" in mime_type:
        main_type, sub_type = mime_type.split("/", 1)
        if sub_type not in ["octet-stream", "binary"]:
            extension = f".{sub_type}"
        elif main_type == "application":
            # For general application types
            if "pdf" in sub_type:
                extension = ".pdf"
            elif "msword" in sub_type:
                extension = ".doc"
            elif "vnd.ms-excel" in sub_type:
                extension = ".xls"
            elif "vnd.ms-powerpoint" in sub_type:
                extension = ".ppt"

    # If the original URL contains an extension, use it
    if "." in url_filename:
        orig_extension = url_filename.split(".")[-1]
        if orig_extension and len(orig_extension) <= 5:  # Reasonable extension length
            extension = f".{orig_extension}"

    # Create filename
    if url_filename and "." in url_filename:
        # Keep original filename and add timestamp (use original extension)
        base_name = url_filename.rsplit(".", 1)[0]
        filename = f"{service_name}_{timestamp}_{base_name}{extension}"
    elif url_filename:
        # Keep original filename and add timestamp
        filename = f"{service_name}_{timestamp}_{url_filename}{extension}"
    else:
        # If filename cannot be obtained from URL
        filename = f"{service_name}_{timestamp}{extension}"

    # Replace special characters
    filename = re.sub(r"[^\w\-\.]", "_", filename)

    # Generate file path
    filepath = os.path.join(DEBUG_FILES_DIR, filename)

    # Save file
    try:
        with open(filepath, "wb") as f:
            f.write(file_data)
        logging.info(
            f"DEBUG: Saved {service_name} file to {filepath} ({len(file_data)} bytes, {mime_type})"
        )
        return filepath
    except Exception as e:
        logging.error(f"DEBUG: Failed to save {service_name} file: {str(e)}")
        return None


def extract_cloud_links(message_content):
    """Extract cloud storage links from the message and return the remaining text and a list of links.

    Args:
        message_content (str): Message content to process

    Returns:
        tuple: (Text with links removed, list of detected cloud storage links)
    """
    # URL patterns for each cloud storage
    dropbox_pattern = r"https?://(?:www\.)?dropbox\.com/\S+"
    gdrive_pattern = r"https?://(?:www\.)?drive\.google\.com/\S+"

    # Combine all patterns (remove OneDrive pattern)
    cloud_pattern = f"({dropbox_pattern}|{gdrive_pattern})"

    # Search for links
    links = re.findall(cloud_pattern, message_content)

    # Create text with links removed
    cleaned_text = re.sub(cloud_pattern, "", message_content).strip()

    return cleaned_text, links


async def download_from_cloud_storage(message, url):
    """Download a file from a cloud storage URL.

    Args:
        message (discord.Message): Discord message object
        url (str): Cloud storage URL

    Returns:
        tuple: (Byte array of downloaded file data, MIME type string)
    """
    # Determine service type from URL
    if "dropbox.com" in url:
        return await download_from_dropbox(message, url)
    elif "drive.google.com" in url:
        return await download_from_google_drive(message, url)
    elif (
        "1drv.ms" in url
        or "onedrive.live.com" in url
        or "sharepoint.com" in url
        or "office.com" in url
        or "microsoft.com" in url
    ):
        await message.add_reaction("❌")
        raise ValueError(
            "OneDriveリンクはサポートされていません。DropboxまたはGoogle Driveをご利用ください。"
        )
    else:
        await message.add_reaction("❌")
        raise ValueError(
            f"サポートされていないクラウドストレージURL: {url}\n現在サポートしているのはDropboxとGoogle Driveのみです。"
        )


async def download_from_dropbox(message, url):
    """Download a file from a Dropbox link.

    Args:
        message (discord.Message): Discord message object
        url (str): Dropbox URL

    Returns:
        tuple: (Byte array of downloaded file data, MIME type string)
    """
    # Convert Dropbox URL to a direct downloadable URL
    # For public links: https://www.dropbox.com/s/xxxx/file.pdf?dl=0 -> https://www.dropbox.com/s/xxxx/file.pdf?dl=1
    download_url = url.replace("?dl=0", "?dl=1")
    if "?dl=" not in download_url:
        download_url = f"{download_url}{'&' if '?' in download_url else '?'}dl=1"

    try:
        async with message.channel.typing():
            await message.add_reaction("☁️")  # Reaction indicating cloud processing

            # Download file
            async with bot.http_session.get(download_url) as resp:
                if resp.status != 200:
                    await message.add_reaction("❌")
                    raise ValueError(
                        f"Failed to download from Dropbox. Status: {resp.status}"
                    )

                file_data = await resp.read()
                mime_type = get_mime_type_from_bytes(file_data)

                # If in debug mode, save the file locally
                if DEBUG_SAVE_CLOUD_FILES:
                    debug_path = save_debug_file(
                        file_data, "dropbox", url, mime_type
                    )
                    if debug_path:
                        await message.add_reaction(
                            "💾"
                        )  # Reaction indicating successful save

                # Reaction indicating download complete
                await message.add_reaction("✅")
                return file_data, mime_type
    except Exception as e:
        await message.add_reaction("❌")
        raise ValueError(f"Error downloading from Dropbox: {str(e)}")


async def download_from_google_drive(message, url):
    """Download a file from a Google Drive link.

    Args:
        message (discord.Message): Discord message object
        url (str): Google Drive URL

    Returns:
        tuple: (Byte array of downloaded file data, MIME type string)
    """
    try:
        async with message.channel.typing():
            await message.add_reaction("☁️")  # Reaction indicating cloud processing

            # Extract Google Drive ID
            file_id = None
            if "/file/d/" in url:
                file_id = url.split("/file/d/")[1].split("/")[0]
            elif "id=" in url:
                file_id = url.split("id=")[1].split("&")[0]
            else:
                await message.add_reaction("❌")
                raise ValueError(
                    f"Unable to extract Google Drive file ID from URL: {url}"
                )

            # Construct direct download URL
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            # Download file
            async with bot.http_session.get(download_url) as resp:
                if resp.status != 200:
                    await message.add_reaction("❌")
                    raise ValueError(
                        f"Failed to download from Google Drive. Status: {resp.status}"
                    )

                # Check cookies to see if a confirmation page for a large file was returned
                if "Content-Disposition" not in resp.headers:
                    # Confirmation token is required for large files
                    body = await resp.text()
                    confirm_token = re.search(r"confirm=([0-9A-Za-z]+)", body)
                    if confirm_token:
                        # Re-download using the confirmation token
                        confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token.group(1)}"
                        async with bot.http_session.get(confirm_url) as confirm_resp:
                            if confirm_resp.status != 200:
                                await message.add_reaction("❌")
                                raise ValueError(
                                    f"Failed to download large file from Google Drive. Status: {confirm_resp.status}"
                                )

                            file_data = await confirm_resp.read()
                    else:
                        # If confirmation token is not found, use the initial response
                        file_data = await resp.read()
                else:
                    # For normal files
                    file_data = await resp.read()

                mime_type = get_mime_type_from_bytes(file_data)

                # If in debug mode, save the file locally
                if DEBUG_SAVE_CLOUD_FILES:
                    debug_path = save_debug_file(
                        file_data, "gdrive", url, mime_type
                    )
                    if debug_path:
                        await message.add_reaction(
                            "💾"
                        )  # Reaction indicating successful save

                # Reaction indicating download complete
                await message.add_reaction("✅")
                return file_data, mime_type
    except Exception as e:
        await message.add_reaction("❌")
        raise ValueError(f"Error downloading from Google Drive: {str(e)}")


async def process_cloud_file(
    message, cleaned_text, file_data, mime_type, save_to_file=False
):
    """Process the file downloaded from cloud storage and generate a response.

    Args:
        message (discord.Message): Discord message object
        cleaned_text (str): Text extracted from the message
        file_data (bytes): File data
        mime_type (str): MIME type of the file
        save_to_file (bool, optional): Whether to save the response as a file
    """
    try:
        # Generate response using Gemini
        files_data = [{"data": file_data, "mime_type": mime_type}]
        response_text = await generate_response_with_files_and_text(
            message, files_data, cleaned_text
        )

        # Branch processing based on !save flag
        if save_to_file:
            await save_response_as_file(message, response_text)
        else:
            await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)
    except Exception as e:
        await message.channel.send(
            f"An error occurred while processing the cloud file: {str(e)}"
        )


async def split_and_send_messages(message_system, text, max_length):
    """Split the message into chunks and send them, respecting the maximum length."""
    start = 0
    while start < len(text):
        if len(text) - start <= max_length:
            await message_system.channel.send(text[start:])
            break

        end = start + max_length
        while end > start and text[end - 1] not in " \n\r\t":
            end -= 1

        if end == start:
            end = start + max_length

        await message_system.channel.send(text[start:end].strip())
        start = end


# Removed old generate_image function
# async def generate_image(prompt_text, aspect_ratio): ...


async def generate_image_session(message, prompt, aspect_ratio):
    """Generates an image using a persistent chat session for multi-turn editing."""
    global image_chat
    user_id = message.author.id

    # Create a new session for !img command
    session = chat_model.aio.chats.create(
        model=GEMINI_IMAGE_MODEL,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio)
        )
    )
    image_chat[user_id] = session

    try:
        response = await session.send_message(prompt)
        return process_image_response(response)
    except Exception as e:
        logging.error("Error in generate_image_session: %s", e, exc_info=True)
        raise

# Helper to process response and extract image
def process_image_response(response):
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
    raise ValueError("No image found in response.")


async def handle_generation(message, prompt, aspect_ratio):
    """Handles the !img command."""
    try:
        parts = [prompt]
        if message.attachments:
             attachment_parts = await download_attachments_as_parts(message)
             parts.extend(attachment_parts)

        image_data = await generate_image_session(message, parts, aspect_ratio)
        
        with io.BytesIO(image_data) as image_binary:
            file = discord.File(image_binary, filename="generated_image.png")
            await message.channel.send(file=file)

        # Sync context to text chat
        asyncio.create_task(update_text_chat_with_image(message, image_data, prompt))
        
    except ValueError as ve:
        await message.channel.send(f"Invalid input or API error: {str(ve)}")
    except Exception as e:
        await message.channel.send(f"Failed to generate image: {str(e)}")

async def handle_edit_generation(message, prompt):
    """Handles the !edit command."""
    global image_chat
    user_id = message.author.id
    session = image_chat.get(user_id)
    
    parts = [prompt]
    if message.attachments:
        attachment_parts = await download_attachments_as_parts(message)
        parts.extend(attachment_parts)

    if not session:
        # Create new session if attachments exist, else error
        if message.attachments:
             session = chat_model.aio.chats.create(
                model=GEMINI_IMAGE_MODEL,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                 )
             )
             image_chat[user_id] = session
        else:
             await message.channel.send("No active image session. Use !img to start one.")
             return

    try:
        response = await session.send_message(parts)
        image_data = process_image_response(response)
        
        with io.BytesIO(image_data) as image_binary:
            file = discord.File(image_binary, filename="edited_image.png")
            await message.channel.send(file=file)

        # Sync context to text chat
        asyncio.create_task(update_text_chat_with_image(message, image_data, prompt))

    except Exception as e:
         await message.channel.send(f"Failed to edit image: {str(e)}")



async def parse_args(args):
    """Parses arguments for image generation.

    Expected format: prompt | aspect_ratio

    Returns:
        (prompt, aspect_ratio, invalid_ratio) where invalid_ratio is None
        when the ratio is supported or omitted, and is the original input
        string otherwise (so the caller can notify the user).
    """
    parts = [part.strip() for part in args.split("|")]
    prompt = parts[0] if len(parts) > 0 else ""
    aspect_ratio = DEFAULT_ASPECT_RATIO
    invalid_ratio = None
    if len(parts) > 1 and parts[1]:
        requested = parts[1]
        if requested in SUPPORTED_ASPECT_RATIOS:
            aspect_ratio = requested
        else:
            invalid_ratio = requested
    return prompt, aspect_ratio, invalid_ratio

async def download_attachments_as_parts(message):
    """Downloads attachments and returns them as a list of types.Part."""
    parts = []
    if message.attachments:
        for attachment in message.attachments:
            try:
                async with bot.http_session.get(attachment.url) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        mime_type = get_mime_type_from_bytes(data)
                        parts.append(types.Part.from_bytes(data=data, mime_type=mime_type))
            except Exception as e:
                logging.error(f"Failed to download attachment {attachment.filename}: {e}")
    return parts


async def update_text_chat_with_image(message, image_data, prompt_text):
    """Updates the text chat session with the generated image context."""
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
        # Construct message content: context info + image
        context_text = f"Context: The user generated/edited an image with the prompt: '{prompt_text}'. The image is attached."
        
        mime_type = get_mime_type_from_bytes(image_data)
        image_part = types.Part.from_bytes(data=image_data, mime_type=mime_type)
        
        # Send to model. We deliberately ignore the response as this is just for context.
        await chat_session.send_message([context_text, image_part])
        logging.info(f"Updated text chat context for user {user_id} with new image.")
        
    except Exception as e:
        logging.error(f"Failed to update text chat context: {e}")


# Run the bot
bot.run(DISCORD_BOT_TOKEN)
