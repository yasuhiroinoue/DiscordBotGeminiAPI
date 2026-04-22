# Gemini DiscordBot using GeminiAPI
import os
import re
import aiohttp
import io

import asyncio
import base64
import magic
import discord
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
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
# Holds strong references to fire-and-forget background tasks so the GC
# cannot cancel them mid-flight (see https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task).
_background_tasks: set[asyncio.Task] = set()


@dataclass
class DeepResearchJob:
    user_id: int
    topic: str
    interaction_id: str
    task: asyncio.Task
    ack_channel_id: int
    ack_message_id: int
    started_at: float  # time.monotonic() at job start


# Per-user running Deep Research jobs. Only one concurrent job per user.
deep_research_jobs: dict[int, DeepResearchJob] = {}


@dataclass
class DeepResearchPlan:
    user_id: int
    topic: str
    plan_interaction_id: str   # latest planning interaction id (chain target)
    plan_text: str              # most recent plan text for display
    channel_id: int             # where the plan was posted
    created_at: float


# Per-user pending plans awaiting an Approve / Refine / Abort decision.
# Invariant: for any user_id, at most one of deep_research_jobs[user_id]
# and deep_research_plans[user_id] is set at any time.
deep_research_plans: dict[int, DeepResearchPlan] = {}

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

# Deep Research (Route B: direct Gemini API key, separate from the Vertex client)
DEEP_RESEARCH_API_KEY = os.getenv("DEEP_RESEARCH_API_KEY")
DEEP_RESEARCH_AGENT = os.getenv("DEEP_RESEARCH_AGENT", "deep-research-preview-04-2026")
DEEP_RESEARCH_MAX_CONCURRENT = int(os.getenv("DEEP_RESEARCH_MAX_CONCURRENT", "2"))
DEEP_RESEARCH_POLL_SECONDS = int(os.getenv("DEEP_RESEARCH_POLL_SECONDS", "20"))
DEEP_RESEARCH_TIMEOUT_SECONDS = int(os.getenv("DEEP_RESEARCH_TIMEOUT_SECONDS", "3900"))
_dr_global_semaphore = asyncio.Semaphore(DEEP_RESEARCH_MAX_CONCURRENT)

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

# Root logging configuration: make logging.info / logging.exception calls
# visible on stderr. Without this, the default root logger level is WARNING
# and our .info diagnostics (Deep Research output shape, grounding metadata,
# etc.) would be silently dropped.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Dedicated access log for user interactions. Does not propagate so it only
# writes to bot_usage.log, not stderr.
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

# Initialize a separate direct-API client for Deep Research, which is not
# currently served over Vertex AI for the preview models.
dr_client = genai.Client(api_key=DEEP_RESEARCH_API_KEY) if DEEP_RESEARCH_API_KEY else None
if dr_client is None:
    print("Deep Research disabled: DEEP_RESEARCH_API_KEY not set.")
else:
    print(f"Deep Research enabled (agent={DEEP_RESEARCH_AGENT}, max_concurrent={DEEP_RESEARCH_MAX_CONCURRENT}).")

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
        dr_command = False

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

        # Detect !dr command (Deep Research). Match both "!dr" and "!dr ..."
        # so a bare "!dr" shows usage instead of falling through to chat and
        # wasting an API call.
        elif cleaned_text == "!dr" or cleaned_text.startswith("!dr "):
            if dr_client is None:
                await message.channel.send(
                    "Deep Research is disabled (DEEP_RESEARCH_API_KEY not set)."
                )
                return
            if message.author.id in deep_research_jobs:
                await message.channel.send(
                    "You already have a Deep Research job running. Wait for it to finish or send `RESET` to cancel."
                )
                return
            if message.author.id in deep_research_plans:
                await message.channel.send(
                    "You already have a pending Deep Research plan. Approve, refine, or abort it first (or send `RESET`)."
                )
                return
            topic = cleaned_text[4:].strip()
            if not topic:
                await message.channel.send("Usage: `!dr <topic>`")
                return
            dr_command = True
            prompt_text = topic

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

            elif dr_command:
                await message.add_reaction("🔬")
                attachments = (
                    await download_attachments_as_parts(message)
                    if message.attachments else []
                )
                view = PlanOrDirectView(
                    user_id=message.author.id,
                    topic=prompt_text,
                    origin_message=message,
                    attachments=attachments,
                )
                ack = await message.channel.send(
                    f"🔬 Deep Research requested: **{prompt_text}**\n"
                    f"Choose a flow below.",
                    view=view,
                )
                view.ack = ack

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
        job = deep_research_jobs.pop(message.author.id, None)
        if job is not None:
            job.task.cancel()
        deep_research_plans.pop(message.author.id, None)
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
    """Saves the response as a markdown file and sends it with an inline preview in a single message."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"gemini_response_{timestamp}.md"

    file = discord.File(io.StringIO(response_text), filename=filename)

    preview_lines = response_text.split("\n")[:5]
    preview = "\n".join(preview_lines)
    if len(preview_lines) >= 5:
        preview += "\n..."

    content = "💾 Here's your response as a file:"
    if preview.strip():
        content += f"\n📝 Preview:\n```\n{preview}\n```"

    await message.channel.send(content, file=file)


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
        task = asyncio.create_task(update_text_chat_with_image(message, image_data, prompt))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

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
        task = asyncio.create_task(update_text_chat_with_image(message, image_data, prompt))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

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


def _build_dr_injection(topic: str, report_text: str) -> str:
    """Build a compact summary (≤1500 chars) of a Deep Research report to feed back
    into the user's regular chat session as a new turn. Uses regex-only extraction
    from the first 2000 chars so it costs nothing extra and stays deterministic.
    """
    head = report_text[:2000]
    bullets: list[str] = []
    for line in head.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r"^#{1,3}\s+(.+)$", stripped)
        if m:
            bullets.append(m.group(1).strip())
        else:
            m = re.match(r"^[-*]\s+(.+)$", stripped)
            if m:
                bullets.append(m.group(1).strip())
        if len(bullets) >= 8:
            break
    bullets_text = "\n".join(f"- {b[:120]}" for b in bullets) or "- (no bullets extracted)"
    injection = (
        f"[Deep Research completed — topic: \"{topic}\"]\n"
        f"Key findings:\n{bullets_text}\n"
        f"The full report has been posted as a .md attachment in this channel.\n"
        f"You may reference it when the user asks follow-up questions."
    )
    return injection[:1500]


async def _inject_dr_summary(message, topic: str, report_text: str) -> None:
    """Append the Deep Research summary as a new turn in the user's text chat session.
    Mirrors update_text_chat_with_image so follow-ups like "summarize the research"
    naturally work via the regular chat path.
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
        await chat_session.send_message(_build_dr_injection(topic, report_text))
        logging.info("Injected Deep Research summary into chat session for user %s", user_id)
    except Exception:
        logging.exception("Failed to inject Deep Research summary for user %s", user_id)


class PlanOrDirectView(discord.ui.View):
    """Initial UI shown after `!dr <topic>`. Picks plan-first vs direct-run."""

    def __init__(
        self,
        *,
        user_id: int,
        topic: str,
        origin_message: discord.Message,
        attachments: list,
        timeout: float = 300,
    ):
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.topic = topic
        self.origin_message = origin_message
        self.attachments = attachments
        self.ack: discord.Message | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This Deep Research request isn't yours.", ephemeral=True
            )
            return False
        return True

    async def on_timeout(self) -> None:
        for child in self.children:
            child.disabled = True
        if self.ack is not None:
            try:
                await self.ack.edit(
                    content=f"⏱️ Deep Research request timed out (no selection)\nTopic: **{self.topic}**",
                    view=self,
                )
            except Exception:
                logging.exception("PlanOrDirectView on_timeout edit failed")

    def _spawn(self, *, planning_mode: bool) -> None:
        task = asyncio.create_task(
            _run_deep_research(
                self.origin_message,
                self.topic,
                self.ack,
                planning_mode=planning_mode,
                attachments=self.attachments,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    @discord.ui.button(label="📋 Plan first", style=discord.ButtonStyle.primary)
    async def plan_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.user_id in deep_research_jobs:
            await interaction.response.send_message(
                "You already have a Deep Research job running.", ephemeral=True
            )
            return
        self.stop()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(
            content=f"📋 Preparing plan for: **{self.topic}**…",
            view=self,
        )
        self._spawn(planning_mode=True)

    @discord.ui.button(label="🚀 Run now", style=discord.ButtonStyle.success)
    async def direct_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if self.user_id in deep_research_jobs:
            await interaction.response.send_message(
                "You already have a Deep Research job running.", ephemeral=True
            )
            return
        self.stop()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(
            content=f"🔬 Deep Research starting: **{self.topic}**…",
            view=self,
        )
        self._spawn(planning_mode=False)


class PlanDecisionView(discord.ui.View):
    """Plan review UI shown after a planning call completes."""

    def __init__(
        self,
        *,
        user_id: int,
        topic: str,
        plan_interaction_id: str,
        origin_message: discord.Message,
        timeout: float = 1800,
    ):
        super().__init__(timeout=timeout)
        self.user_id = user_id
        self.topic = topic
        self.plan_interaction_id = plan_interaction_id
        self.origin_message = origin_message
        self.ack: discord.Message | None = None

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message(
                "This plan isn't yours.", ephemeral=True
            )
            return False
        return True

    def _plan_is_fresh(self) -> bool:
        current = deep_research_plans.get(self.user_id)
        return current is not None and current.plan_interaction_id == self.plan_interaction_id

    async def on_timeout(self) -> None:
        for child in self.children:
            child.disabled = True
        if self.ack is not None:
            try:
                await self.ack.edit(content="⏱️ Plan decision timed out.", view=self)
            except Exception:
                logging.exception("PlanDecisionView on_timeout edit failed")

    @discord.ui.button(label="✅ Approve & execute", style=discord.ButtonStyle.success)
    async def approve_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._plan_is_fresh():
            await interaction.response.send_message(
                "This plan has been reset. Please start a new `!dr` request.", ephemeral=True
            )
            return
        if self.user_id in deep_research_jobs:
            await interaction.response.send_message(
                "You already have a Deep Research job running.", ephemeral=True
            )
            return
        deep_research_plans.pop(self.user_id, None)
        self.stop()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(
            content=f"✅ Approved — executing research on **{self.topic}**",
            view=self,
        )
        exec_ack = await interaction.channel.send(
            f"🔬 Deep Research starting (execute): **{self.topic}**\nStatus: queued…"
        )
        task = asyncio.create_task(
            _run_deep_research(
                self.origin_message,
                self.topic,
                exec_ack,
                planning_mode=False,
                previous_interaction_id=self.plan_interaction_id,
                input_override="Execute the plan.",
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    @discord.ui.button(label="✏️ Refine", style=discord.ButtonStyle.primary)
    async def refine_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._plan_is_fresh():
            await interaction.response.send_message(
                "This plan has been reset. Please start a new `!dr` request.", ephemeral=True
            )
            return
        if self.user_id in deep_research_jobs:
            await interaction.response.send_message(
                "You already have a Deep Research job running.", ephemeral=True
            )
            return
        await interaction.response.send_modal(
            RefineModal(
                user_id=self.user_id,
                topic=self.topic,
                plan_interaction_id=self.plan_interaction_id,
                origin_message=self.origin_message,
                view=self,
            )
        )

    @discord.ui.button(label="🛑 Abort", style=discord.ButtonStyle.danger)
    async def abort_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        deep_research_plans.pop(self.user_id, None)
        self.stop()
        for child in self.children:
            child.disabled = True
        await interaction.response.edit_message(
            content=f"🛑 Plan aborted: **{self.topic}**",
            view=self,
        )


class RefineModal(discord.ui.Modal, title="Refine Deep Research Plan"):
    refinement = discord.ui.TextInput(
        label="Refinement instructions",
        style=discord.TextStyle.paragraph,
        placeholder="e.g., Include more on cost and power consumption.",
        max_length=2000,
        required=True,
    )

    def __init__(
        self,
        *,
        user_id: int,
        topic: str,
        plan_interaction_id: str,
        origin_message: discord.Message,
        view: "PlanDecisionView",
    ):
        super().__init__()
        self.user_id = user_id
        self.topic = topic
        self.plan_interaction_id = plan_interaction_id
        self.origin_message = origin_message
        self.parent_view = view

    async def on_submit(self, interaction: discord.Interaction):
        text = self.refinement.value.strip()
        if not text:
            await interaction.response.send_message("Empty refinement ignored.", ephemeral=True)
            return
        if self.user_id in deep_research_jobs:
            await interaction.response.send_message(
                "You already have a Deep Research job running.", ephemeral=True
            )
            return
        current = deep_research_plans.get(self.user_id)
        if current is None or current.plan_interaction_id != self.plan_interaction_id:
            await interaction.response.send_message(
                "This plan has been reset. Please start a new `!dr` request.", ephemeral=True
            )
            return

        await interaction.response.send_message("✏️ Refining plan…", ephemeral=True)

        # Clear the prior plan; a new one will replace it when refinement completes.
        deep_research_plans.pop(self.user_id, None)
        self.parent_view.stop()
        for child in self.parent_view.children:
            child.disabled = True
        if self.parent_view.ack is not None:
            try:
                await self.parent_view.ack.edit(view=self.parent_view)
            except Exception:
                logging.exception("Failed to disable old PlanDecisionView on refine")

        refine_ack = await interaction.channel.send(
            f"📋 Refining plan with: {text[:200]}\nTopic: **{self.topic}**"
        )
        task = asyncio.create_task(
            _run_deep_research(
                self.origin_message,
                self.topic,
                refine_ack,
                planning_mode=True,
                previous_interaction_id=self.plan_interaction_id,
                input_override=text,
            )
        )
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)


async def _run_deep_research(
    message,
    topic: str,
    ack,
    *,
    planning_mode: bool = False,
    previous_interaction_id: str | None = None,
    input_override: str | None = None,
    attachments: list | None = None,
) -> None:
    """Fire-and-forget task that drives a Deep Research interaction to completion.

    Modes:
    - planning_mode=True: run a collaborative-planning call. On completion,
      store the plan in deep_research_plans and attach a PlanDecisionView.
    - planning_mode=False (default): run execution. May chain from a previous
      plan via previous_interaction_id. On completion, post the .md report,
      inject a compact summary into the text chat session, and post any
      generated visualization images.
    """
    user_id = message.author.id
    started = time.monotonic()
    last_heartbeat = 0.0
    interaction_id: str | None = None
    ack_prefix = "📋 Planning" if planning_mode else "🔬 Deep Research"
    try:
        async with _dr_global_semaphore:
            agent_config: dict = {"visualization": "auto"}
            if planning_mode:
                agent_config["collaborative_planning"] = True

            if previous_interaction_id is None:
                effective_input = (
                    [topic, *attachments] if attachments else topic
                )
            else:
                effective_input = input_override or "Execute the plan."

            create_kwargs = dict(
                agent=DEEP_RESEARCH_AGENT,
                input=effective_input,
                agent_config=agent_config,
                background=True,
                store=True,
            )
            if previous_interaction_id is not None:
                create_kwargs["previous_interaction_id"] = previous_interaction_id

            try:
                interaction = await asyncio.to_thread(
                    dr_client.interactions.create, **create_kwargs
                )
            except Exception as e:
                logging.exception(
                    "Deep Research create failed for user %s (planning_mode=%s)",
                    user_id, planning_mode,
                )
                try:
                    await ack.edit(content=f"❌ Failed to start Deep Research: {e}", view=None)
                except Exception:
                    logging.exception("Failed to edit ack on create failure")
                if not planning_mode:
                    await message.add_reaction("❌")
                return

            interaction_id = interaction.id
            deep_research_jobs[user_id] = DeepResearchJob(
                user_id=user_id,
                topic=topic,
                interaction_id=interaction_id,
                task=asyncio.current_task(),
                ack_channel_id=ack.channel.id,
                ack_message_id=ack.id,
                started_at=started,
            )

            await ack.edit(
                content=(
                    f"{ack_prefix} running\n"
                    f"Topic: **{topic}**\n"
                    f"id: `{interaction_id}`\n"
                    f"elapsed 0m, status={interaction.status}"
                ),
                view=None,
            )
            last_heartbeat = time.monotonic()

            while True:
                elapsed = time.monotonic() - started
                if elapsed > DEEP_RESEARCH_TIMEOUT_SECONDS:
                    await ack.edit(
                        content=(
                            f"⏱️ {ack_prefix} timed out after {int(elapsed / 60)}m\n"
                            f"Topic: **{topic}**\n"
                            f"id: `{interaction_id}`"
                        ),
                        view=None,
                    )
                    if not planning_mode:
                        await message.add_reaction("❌")
                    return

                await asyncio.sleep(DEEP_RESEARCH_POLL_SECONDS)

                try:
                    interaction = await asyncio.to_thread(
                        dr_client.interactions.get, interaction_id
                    )
                except Exception as e:
                    logging.exception(
                        "Deep Research poll failed for user %s (id=%s)",
                        user_id, interaction_id,
                    )
                    try:
                        await ack.edit(
                            content=f"❌ Deep Research poll failed: {e}", view=None
                        )
                    except Exception:
                        logging.exception("Failed to edit ack on poll failure")
                    if not planning_mode:
                        await message.add_reaction("❌")
                    return

                if interaction.status in ("completed", "failed"):
                    break

                if time.monotonic() - last_heartbeat >= 60:
                    elapsed_min = int((time.monotonic() - started) / 60)
                    try:
                        await ack.edit(
                            content=(
                                f"{ack_prefix} running\n"
                                f"Topic: **{topic}**\n"
                                f"id: `{interaction_id}`\n"
                                f"elapsed {elapsed_min}m, status={interaction.status}"
                            ),
                            view=None,
                        )
                    except Exception:
                        logging.exception("Failed to edit heartbeat for user %s", user_id)
                    last_heartbeat = time.monotonic()

            elapsed_min = int((time.monotonic() - started) / 60)

            if interaction.status == "completed":
                try:
                    outputs = interaction.outputs or []
                    logging.info(
                        "Deep Research completed for user %s (planning_mode=%s): %d outputs, types=%s",
                        user_id, planning_mode, len(outputs),
                        [getattr(o, "type", "?") for o in outputs],
                    )
                    text_parts: list[str] = []
                    image_files: list[discord.File] = []
                    for i, out in enumerate(outputs):
                        typ = getattr(out, "type", None)
                        if typ == "image":
                            raw = getattr(out, "data", None)
                            if raw:
                                try:
                                    img_bytes = (
                                        base64.b64decode(raw)
                                        if isinstance(raw, str) else bytes(raw)
                                    )
                                    image_files.append(
                                        discord.File(
                                            io.BytesIO(img_bytes),
                                            filename=f"dr_viz_{i}.png",
                                        )
                                    )
                                except Exception:
                                    logging.exception(
                                        "Failed to decode Deep Research image %d", i
                                    )
                        else:
                            txt = getattr(out, "text", None)
                            if txt:
                                text_parts.append(txt)
                    result_text = "\n\n".join(text_parts).strip()
                except Exception:
                    result_text = ""
                    image_files = []
                    logging.exception("Failed to extract result for user %s", user_id)

                if not result_text:
                    await ack.edit(
                        content=(
                            f"⚠️ {ack_prefix} completed but returned empty output\n"
                            f"Topic: **{topic}**\n"
                            f"id: `{interaction_id}`"
                        ),
                        view=None,
                    )
                    if not planning_mode:
                        await message.add_reaction("⚠️")
                    return

                if planning_mode:
                    deep_research_plans[user_id] = DeepResearchPlan(
                        user_id=user_id,
                        topic=topic,
                        plan_interaction_id=interaction_id,
                        plan_text=result_text,
                        channel_id=ack.channel.id,
                        created_at=started,
                    )
                    header = f"📋 Deep Research plan ({elapsed_min}m) — topic: **{topic}**"
                    combined = f"{header}\n\n{result_text}"
                    if len(combined) <= MAX_DISCORD_LENGTH:
                        await message.channel.send(combined)
                    else:
                        await message.channel.send(header)
                        await split_and_send_messages(
                            message, result_text, MAX_DISCORD_LENGTH
                        )
                    decision_view = PlanDecisionView(
                        user_id=user_id,
                        topic=topic,
                        plan_interaction_id=interaction_id,
                        origin_message=message,
                    )
                    decision_ack = await message.channel.send(
                        "Approve, refine, or abort this plan:",
                        view=decision_view,
                    )
                    decision_view.ack = decision_ack
                    await ack.edit(
                        content=(
                            f"📋 Plan ready ({elapsed_min}m)\n"
                            f"Topic: **{topic}**\n"
                            f"id: `{interaction_id}`"
                        ),
                        view=None,
                    )
                    # No ✅ reaction yet — only the final execute gets it.
                else:
                    await save_response_as_file(message, result_text)
                    await message.channel.send(
                        f"✅ Deep Research completed ({elapsed_min}m) — topic: **{topic}**"
                    )
                    await _inject_dr_summary(message, topic, result_text)
                    for chunk_start in range(0, len(image_files), 10):
                        chunk = image_files[chunk_start:chunk_start + 10]
                        header = (
                            "🖼️ Visualizations"
                            if len(image_files) <= 10
                            else f"🖼️ Visualizations ({chunk_start + 1}-{chunk_start + len(chunk)} of {len(image_files)})"
                        )
                        await message.channel.send(content=header, files=chunk)
                    await message.add_reaction("✅")
                    await ack.edit(
                        content=(
                            f"✅ Deep Research completed ({elapsed_min}m)\n"
                            f"Topic: **{topic}**\n"
                            f"id: `{interaction_id}`"
                        ),
                        view=None,
                    )
            else:  # failed
                error_msg = getattr(interaction, "error", "<unknown>")
                await message.channel.send(f"❌ {ack_prefix} failed: {error_msg}")
                await ack.edit(
                    content=(
                        f"❌ {ack_prefix} failed ({elapsed_min}m)\n"
                        f"Topic: **{topic}**\n"
                        f"id: `{interaction_id}`\n"
                        f"error: {error_msg}"
                    ),
                    view=None,
                )
                if not planning_mode:
                    await message.add_reaction("❌")

    except asyncio.CancelledError:
        try:
            await ack.edit(
                content=f"🛑 {ack_prefix} cancelled\nTopic: **{topic}**", view=None
            )
        except Exception:
            logging.exception("Failed to edit ack on cancel")
        raise
    except Exception:
        logging.exception("Deep Research task failed unexpectedly for user %s", user_id)
        try:
            await ack.edit(
                content=f"❌ {ack_prefix} encountered an unexpected error\nTopic: **{topic}**",
                view=None,
            )
        except Exception:
            logging.exception("Failed to edit ack on unexpected error")
    finally:
        deep_research_jobs.pop(user_id, None)


# Run the bot
bot.run(DISCORD_BOT_TOKEN)
