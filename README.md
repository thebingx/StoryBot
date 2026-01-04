# AI Story Kingdom

An interactive web application that generates AI-powered children's stories with text-to-speech (TTS) narration and illustrations. Built with FastAPI, featuring Mario-themed UI and offline TTS capabilities.

## Features

- **Story Generation**: Create engaging stories for kids using AI (OpenAI API)
- **Text-to-Speech**: Offline TTS using Kokoro models with multiple voices
- **Image Generation**: AI-generated illustrations for stories
- **Web Interface**: Mario-themed UI with difficulty levels and character customization
- **Story Library**: Save and replay generated stories
- **Offline Mode**: Run TTS completely offline after initial setup

## Prerequisites

- Python 3.8+
- Git
- Internet connection (for initial setup and API calls)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/StoryBot.git
   cd StoryBot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r server/requirements.txt
   ```

## Configuration

### Environment Variables (.env)

Create a `.env` file in the root directory with your API keys:

```env
MIMO_API_KEY=your_mimo_api_key_here
VOLC_API_KEY=your_volc_api_key_here
```

- **MIMO_API_KEY**: Required for story and image prompt generation (from Xiaomimimo API)
- **VOLC_API_KEY**: Required for image generation (Volcengine Ark API)

**Important**: Never commit your `.env` file to version control. It's already in `.gitignore`.

## Running the Application

### Online Mode (Default)

Start the server with internet access for full functionality:

```bash
python server/main.py
```

The app will be available at `http://localhost:8000`

### Offline TTS Mode

To run the TTS server completely offline (after initial model download):

1. Set the Hugging Face offline environment variable
2. Start the server

```bash
HF_HUB_OFFLINE=1 python server/main.py
```

**Note**: On first run, models and voices will be downloaded. Subsequent runs can be offline if `HF_HUB_OFFLINE=1` is set.

## Usage

1. Open `http://localhost:8000` in your browser
2. Enter a story topic (e.g., "A flying dinosaur")
3. Select difficulty level (Easy, Normal, Hard)
4. Optionally customize character name and specific words
5. Click "START ADVENTURE"
6. Listen to the story with TTS narration
7. View generated illustrations
8. Save stories to your library at `/storybook`

## API Endpoints

- `GET /` - Main application page
- `POST /generate-story` - Generate a new story (streaming response)
- `POST /tts` - Convert text to speech audio
- `GET /storybook` - Story library page
- `GET /api/stories` - Get all saved stories
- `DELETE /api/stories/{id}` - Delete a saved story

## Project Structure

```
StoryBot/
├── index.html              # Main web interface
├── storybook.html          # Story library interface
├── server/
│   ├── main.py            # FastAPI server
│   ├── models.py          # TTS model handling
│   ├── requirements.txt   # Python dependencies
│   ├── config.json        # TTS configuration (auto-downloaded)
│   ├── kokoro-v1_0.pth    # TTS model (auto-downloaded)
│   └── voices/            # Voice models (auto-downloaded)
├── data/                  # Saved stories and images
├── .env                   # Environment variables (create this)
└── .gitignore            # Git ignore rules
```

## Development

- The app uses FastAPI for the backend
- Frontend is vanilla HTML/CSS/JavaScript
- TTS powered by Kokoro (offline capable)
- Story generation via OpenAI-compatible API
- Images generated via Volcengine Ark API

## Troubleshooting

- **TTS not working**: Ensure models are downloaded (run once online)
- **API errors**: Check your `.env` file and API keys
- **Port conflicts**: Change port in `main.py` if 8000 is in use
- **Offline mode issues**: Make sure `HF_HUB_OFFLINE=1` is set before starting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

MIT License - see LICENSE file for details