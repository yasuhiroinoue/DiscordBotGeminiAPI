# Gemini Discord Bot

Gemini Discord Bot allows you to converse on Discord using Google's Gemini API. It supports text-based conversations and responds to attached files.

## Features

- **Text-based Conversation**: Utilizes the Gemini API for natural language conversations.
- **Image Recognition**: Recognizes and responds to the content of uploaded images.
- **File Recognition**: Analyzes and responds to the content of uploaded files.
- **Conversation History**: Maintains user-specific conversation history for context-aware responses.
- **History Reset**: Resets the conversation history when the user sends `RESET`.
- **Discord Mentions and Special Characters Handling**: Processes mentions and special characters appropriately and sends clean text to the Gemini API.
- **Long Message Splitting**: Splits messages exceeding Discord's 2000-character limit into smaller chunks for seamless transmission.
- **Google Search Tool**: Generates responses based on Google Search results.
- **Cloud File Download**: If a message contains shared links from Dropbox or Google Drive, the bot will download the linked files and include their content when sending the message to Gemini.
- **Multimodal Support**: Attach multiple files (images, PDFs, etc.) in a single message for analysis.
- **Image Generation (Gemini 2.5 Flash)**: Generates high-quality images using the Gemini 2.5 Flash Image model.
- **Multi-turn Image Editing**: Edit generated images conversationally using `!edit`.
- **Context Synchronization**: Automatically shares generated/edited images with the main chat, allowing for immediate follow-up questions (e.g., "What is in this image?").

---

## Setup

1. **Create a Discord Bot**:
   - Create a bot on the [Discord Developer Portal](https://discord.com/developers/applications) and obtain the Bot Token.

2. **Obtain a Google AI Studio API Key**:
   - Get an API Key from [Google AI Studio](https://makersuite.google.com/).
   - Alternatively, if using Vertex AI API, configure your Google Cloud Platform project credentials.

3. **Set Environment Variables**:
   - Create a `.env` file and set the following variables:

     ```env
     DISCORD_BOT_TOKEN=Your Discord Bot Token
     MODEL_ID="gemini-3-pro-preview" # or your preferred model
     # IMAGEN_MODEL="imagen-3.0-generate-001" # Deprecated
     GEMINI_IMAGE_MODEL="gemini-2.5-flash-image" # New image generation model
     
     # Create at https://makersuite.google.com/
     GOOGLE_AI_KEY=Your Google AI Studio API Key
     
     # GCP_PROJECT_ID=Your Google Cloud Platform Project ID # Required for Vertex AI
     # GCP_REGION=Your Google Cloud Platform Region # Required for Vertex AI
     
     IMG_COMMANDS_ENABLED=True/False # Enable/Disable image generation commands (default: False)
     ALLOWED_USER_IDS=yyyyyyyyyyxxxxx,yyyyyyyyyyyyyyy # Comma-separated list of Discord User IDs
     DEBUG_SAVE_CLOUD_FILES=True/False # Enable/Disable saving of downloaded cloud files for debugging (default: False)
     DEBUG_LOG_USER_IDS=True/False # Enable/Disable logging of user IDs for debugging (default: False)
     ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Tools Configuration**:

   To configure the `generate_content_config` in your code, follow these guidelines:

   - The `tools` parameter is included by default in the configuration:
     ```python
     generate_content_config = types.GenerateContentConfig(
         temperature=1,
         top_p=0.95,
         max_output_tokens=8192,
         tools=tools,  # Comment out this line if you are using gemini-2.0-flash-thinking-exp
         # safety_settings=[...],  # Example safety settings (optional)
     )
     ```

   - **Important Note**: When using `gemini-2.0-flash-thinking-exp`, ensure that the `tools` parameter is commented out. This configuration ensures compatibility with the specific version.

7. **Run the Bot**:
   ```bash
   python GeminiDiscordBot.py
   ```

---

## Usage

- **Text Conversation**: Mention the bot or send a direct message (DM) to start a conversation.
- **File Recognition**: Upload a file, with or without accompanying text, and the bot will analyze and respond to its content.
- **Reset Conversation History**: Send `RESET` to clear the conversation history.
- **Image Generation**: Use the `!img` command to start a new image generation session.
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
