# Gemini Discord Bot

Gemini Discord Bot is a bot that allows you to converse on Discord using Google's Gemini Pro API. It supports not only text-based conversations but also understanding and responding to an attached file.

## Features

*   **Text-based Conversation**: Utilizes the Gemini Pro API for natural language conversations.
*   **Image Recognition**: Recognizes and responds to the content of uploaded images.
*   **Conversation History**: Maintains conversation history per user for context-aware responses.
*   **History Reset**:  Resets the conversation history when a user sends `RESET`.
*   **Discord Mentions and Special Characters Handling**: Properly processes Discord mentions and special characters, sending clean text to the Gemini Pro API.
*   **Long Message Splitting**: Splits messages exceeding Discord's character limit (2000 characters) into smaller chunks for seamless transmission.

## Setup

1. **Create a Discord Bot**:
    *   Create a bot on the Discord Developer Portal and obtain the Bot Token.
    *   [Discord Developer Portal](https://discord.com/developers/applications)
2. **Obtain a Google AI Studio API Key**:
    *   Get an API Key from Google AI Studio.
    *   [Google AI Studio](https://makersuite.google.com/)
3. **Set Environment Variables**:
    *   Create a `.env` file and set the following environment variables:
        DISCORD_BOT_TOKEN=Your Discord Bot Token
        GOOGLE_AI_KEY=Your Google AI Studio API Key
4. **Install Dependencies**:
    pip install -r requirements.txt
5. **Run the Bot**:
    python GeminiDiscordBot.py

## Usage

*   **Text Conversation**: Mention the bot or send a direct message (DM) to the bot to start a conversation.
*   **File Recognition**: Upload a file with or without accompanying text, and the bot will analyze and respond to its content.
*   **Reset Conversation History**: Send `RESET` to clear the conversation history.

## Important Notes

*   The Google AI Studio API has usage limits and may incur charges beyond the free tier. Please refer to the Google AI Studio documentation for pricing details.
*   Please use this bot in compliance with the Gemini Pro API Terms of Service.
*   Unexpected errors or inappropriate responses may occur. Please use it with understanding.
*   If you get error when installing `libmagic`
    *   MacOS: `brew install libmagic`
    *   Linux(Debian/Ubuntu): `sudo apt-get install libmagic1`
    *   Windows: Download `libmagic` from the official website and add it to the environment variables.

## Disclaimer

*   The developer is not responsible for any damages caused by the use of this bot.
*   This bot utilizes the Gemini Pro API, and the developer is not responsible for the content of its responses.

## License

MIT License
