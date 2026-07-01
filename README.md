# Gemini Discord Bot

Gemini Discord Bot allows you to converse on Discord using Google's Gemini API. It supports text-based conversations and responds to attached files.

## Features

- **Text-based Conversation**: Utilizes the Gemini API for natural language conversations.
- **Image & File Recognition**: Analyzes and responds to the content of uploaded images and files.
- **Multimodal Support**: Attach multiple files (images, PDFs, etc.) in a single message for analysis.
- **Conversation History**: Maintains user-specific conversation history for context-aware responses.
- **History Reset**: Resets the conversation history (and cancels running Deep Research jobs) when the user sends `RESET`.
- **Discord Mentions and Special Characters Handling**: Processes mentions and special characters appropriately and sends clean text to the Gemini API.
- **Markdown-safe Message Splitting**: Splits responses over Discord's 2000-character limit without breaking fenced code blocks (the closing fence is kept balanced across messages and code indentation is preserved).
- **Google Search Tool**: Grounds responses with Google Search results.
- **Cloud File Download**: If a message contains shared links from Dropbox or Google Drive, the bot downloads the linked files and includes their content when sending the message to Gemini.
- **Save to File (`!save`)**: Returns the response as a Markdown file attachment (useful for long answers); attachments can be sent alongside.
- **Deep Research (`!dr`)**: Runs Google's Gemini Deep Research agent in the background — plan / refine / execute via Discord buttons, multimodal input, and automatic chart visualizations. Requires a separate direct API key (see Setup).
- **Image Generation (`!img`)**: Generates images with the configured Gemini image model (`GEMINI_IMAGE_MODEL`). Disabled unless `IMG_COMMANDS_ENABLED=true`.
- **Multi-turn Image Editing (`!edit`)**: Edit the last generated image conversationally within the same session.
- **Context Synchronization**: Automatically shares generated/edited images with the main chat, allowing immediate follow-up questions (e.g., "What is in this image?").
- **In-Discord Help (`!help`)**: Shows a Japanese command reference that reflects the current feature toggles.

---

## Setup

1. **Create a Discord Bot**:
   - Create a bot on the [Discord Developer Portal](https://discord.com/developers/applications) and obtain the Bot Token.

2. **Set up Google Gemini access**:
   - **Normal chat / vision / image commands (default): Vertex AI.** The bot creates its main client with `genai.Client(vertexai=True, ...)`, so configure a Google Cloud project (`GCP_PROJECT_ID`, `GCP_REGION`) and authenticate (e.g. `gcloud auth application-default login`). `GOOGLE_AI_KEY` is **not** required for this default setup — it only applies if you switch the main client to the direct Gemini API (the commented-out client in `GeminiDiscordBot.py`).
   - **Deep Research (`!dr`): direct Gemini API key.** The Deep Research preview models are not served over Vertex AI, so `!dr` needs a separate `DEEP_RESEARCH_API_KEY` from [Google AI Studio](https://aistudio.google.com/). Leave it empty to disable `!dr`.

3. **Set Environment Variables**:
   - The simplest path is to copy the provided sample and edit it — `env.sample` is the source of truth and lists every variable with its default:

     ```bash
     cp env.sample .env
     ```

   - Key variables:

     ```env
     # --- Required ---
     DISCORD_BOT_TOKEN="your-discord-bot-token"
     GCP_PROJECT_ID="your-gcp-project-id"   # Vertex AI (normal chat)
     GCP_REGION="us-central1"               # Vertex AI region

     # --- Models ---
     MODEL_ID="gemini-3.1-pro-preview"
     GEMINI_IMAGE_MODEL="gemini-3-pro-image-preview"

     # --- Access control (strongly recommended) ---
     ALLOWED_USER_IDS=""   # Comma-separated Discord user IDs; empty = everyone can use the bot

     # --- Optional features ---
     IMG_COMMANDS_ENABLED="false"   # Enable !img / !edit
     DEEP_RESEARCH_API_KEY=""       # Direct Gemini API key; enables !dr (empty = disabled)
     # DEEP_RESEARCH_AGENT="deep-research-preview-04-2026"   # or deep-research-max-preview-04-2026
     # DEEP_RESEARCH_MAX_CONCURRENT="2"
     # DEEP_RESEARCH_POLL_SECONDS="20"
     # DEEP_RESEARCH_TIMEOUT_SECONDS="3900"

     # --- Debug ---
     DEBUG_SAVE_CLOUD_FILES="false"
     # DEBUG_FILES_DIR="debug_files"
     DEBUG_LOG_USER_IDS="false"

     # GOOGLE_AI_KEY=""   # Legacy/optional; not used by the default Vertex AI setup
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Model Configuration** (optional):

   Normal chat uses the `generate_content_config` defined in `GeminiDiscordBot.py`. Its current defaults are:
   ```python
   generate_content_config = types.GenerateContentConfig(
       temperature=1,
       top_p=0.95,
       max_output_tokens=65535,
       safety_settings=[  # all four categories set to "OFF" — see Security Considerations
           types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
           types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
           types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
           types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
       ],
       tools=[types.Tool(google_search=types.GoogleSearch())],  # Google Search grounding
       response_modalities=["TEXT"],
   )
   ```
   Adjust these values in code if needed. Note the safety thresholds default to `OFF` — see [Security Considerations](#security-considerations) before deploying.

6. **Run the Bot**:
   ```bash
   python GeminiDiscordBot.py
   ```

---

## Usage

- **Text Conversation**: Mention the bot or send a direct message (DM) to start a conversation.
- **File Recognition**: Upload a file, with or without accompanying text, and the bot will analyze and respond to its content.
- **Save Response as File**: Prefix your message with `!save ` to receive the answer as a Markdown file attachment instead of inline text (handy for long responses). Attachments can be included.
  ```text
  !save Summarize this PDF in detail
  ```
- **Reset Conversation History**: Send `RESET` to clear the conversation history (this also cancels any running Deep Research job and clears image history).
- **Help**: Send `!help` to show the Japanese command reference inside Discord. The message reflects the current feature toggles (`IMG_COMMANDS_ENABLED`, `DEEP_RESEARCH_API_KEY`) so disabled commands are marked accordingly.
- **Image Generation**: Use the `!img` command to start a new image generation session (requires `IMG_COMMANDS_ENABLED=true`).
  ```text
  !img <prompt> | <aspect_ratio>
  ```
  - **Prompt**: Describe the image to generate.
  - **Aspect Ratio** (optional): Specify aspect ratio (e.g., `16:9`, `1:1`). Default is `1:1`.
  - **Attachments**: You can attach an image to use as a reference.

  Example:
  ```text
  !img a futuristic city | 16:9
  ```

- **Image Editing**: Use the `!edit` command to modify the last generated image.
  ```text
  !edit <instruction>
  ```
  - **Instruction**: Describe how to change the image (e.g., "Make it night", "Add a flying car").
  - The context is preserved, allowing for continuous iteration.

- **Multimodal Chat**:
  - After generating or editing an image, it is automatically shared with the text chat.
  - You can immediately ask questions about the image in the normal chat (e.g., "Describe the architecture").

- **Deep Research**: Use the `!dr` command to run Google's Gemini Deep Research agent in the background.
  ```text
  !dr <topic>
  ```
  After you send `!dr <topic>`, the bot posts two buttons:
    - **📋 Plan first** — the agent returns a short research plan. You can then click **✅ Approve & execute**, **✏️ Refine** (opens a modal for refinement instructions), or **🛑 Abort**. Refining generates a new plan that you can iterate on again.
    - **🚀 Run now** — skip planning and execute the research immediately (previous behavior).
  - **Topic**: The research question. The agent performs multi-step web search and synthesis.
  - **Attachments (multimodal input)**: Attach an image or PDF to your `!dr` message to give the agent visual context. The attachments are re-sent across the plan → refine → approve chain (not just the first call), so the multimodal context is preserved throughout the flow.
  - **Visualizations**: The agent may embed PNG charts automatically when the topic asks for them (e.g., "include a chart comparing market share"). Generated images are posted alongside the report file. No configuration needed.
  - **Latency**: Several minutes to tens of minutes per job (60-minute hard cap). Normal chat remains fully responsive while a research job runs.
  - **Cost**: Deep Research is a paid, compute-heavy feature — cost varies by model and usage and can be significant per run. Set `ALLOWED_USER_IDS` and keep `DEEP_RESEARCH_MAX_CONCURRENT` modest.
  - **Output**: The full report is delivered as a Markdown attachment. A compact summary is automatically injected into the text chat session so you can ask follow-ups like "summarize the key findings" or "what sources did it cite?" without re-uploading anything.
  - **Cancel**: Send `RESET` to cancel a running job and clear any pending plan (also clears the chat and image history).
  - **Requires**: `DEEP_RESEARCH_API_KEY` environment variable (Google AI Studio key). The Vertex AI client used for normal chat does not currently serve the Deep Research preview models, so a separate direct API key is required.

---

## Security Considerations

**Safety filters are disabled by default.** In `GeminiDiscordBot.py`, all four Gemini safety categories (`HARM_CATEGORY_HATE_SPEECH`, `HARM_CATEGORY_DANGEROUS_CONTENT`, `HARM_CATEGORY_SEXUALLY_EXPLICIT`, `HARM_CATEGORY_HARASSMENT`) are configured with `threshold="OFF"` in `generate_content_config`. Combined with an open access list, this lets anyone who can message the bot generate content that would otherwise be filtered.

- **Always set `ALLOWED_USER_IDS` for any non-private deployment.** When it is empty, the startup log prints `Warning: No ALLOWED_USER_IDS set. Bot will respond to everyone.` — treat this as a deployment blocker unless the bot is truly private.
- To re-enable moderation, edit `generate_content_config` in `GeminiDiscordBot.py` and raise the thresholds (for example, `threshold="BLOCK_MEDIUM_AND_ABOVE"`). Refer to the [Gemini API safety settings documentation](https://ai.google.dev/gemini-api/docs/safety-settings) for the available values.

---

## Important Notes

- The Google AI Studio API has usage limits and may incur charges beyond the free tier. Refer to the [Google AI Studio documentation](https://makersuite.google.com/) for pricing details.
- Use this bot in compliance with the Gemini API Terms of Service.
- Unexpected errors or inappropriate responses may occur. Please use it with discretion.
- If you encounter errors installing `libmagic`:
  - **MacOS**: `brew install libmagic`
  - **Linux (Debian/Ubuntu)**: `sudo apt-get install libmagic1`
  - **Windows**: Download `libmagic` from the official website and add it to your environment variables.

---

## Disclaimer

- The developer is not responsible for any damages caused by the use of this bot.
- This bot utilizes the Gemini API, and the developer is not responsible for the content of its responses.

---

## License

This project is licensed under the MIT License.
