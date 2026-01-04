import json
import os
import requests
from datetime import datetime
import asyncio
import io
import torch
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import AsyncOpenAI
from dotenv import load_dotenv
import httpx # Recommendation: use httpx for async API calls instead of requestsimport asyncio
import random

# Import the model builder from your local models.py (assumed to be in /server)
from models import build_model

# 1. SETUP PATHS & ENV
# Load .env from the root folder (one level up)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

app = FastAPI()

# 2. CORS CONFIGURATION
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. MODELS & CLIENTS INITIALIZATION
MOCK_MODE = True # Set to False to use the real Ark API
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'kokoro-v1_0.pth' # Relative to /server
VOICE_PATH = 'voices/af_nova.pt' # Relative to /server

# Initialize MiMo Client
client = AsyncOpenAI(
    api_key=os.environ.get("MIMO_API_KEY"),
    base_url="https://api.xiaomimimo.com/v1"
)

# Set environment variable to offline mode for TTS
os.environ["HF_HUB_OFFLINE"] = "1"

print(f"Loading Kokoro TTS on {DEVICE}...")
# Note: Ensure the model files are inside the /server folder
model = build_model(MODEL_PATH, DEVICE)

# 4. SCHEMAS
class StoryRequest(BaseModel):
    topic: str
    level: str = "easy"
    character_name: str = "Random"
    specific_words: str = ""

class TTSRequest(BaseModel):
    text: str

# 5. HELPER FUNCTIONS
def save_story(story_data):
    # 1. Setup Directories
    data_dir = os.path.join(BASE_DIR, "data")
    image_dir = os.path.join(data_dir, "images") # Folder for images
    
    for folder in [data_dir, image_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 2. Add Timestamp
    # Saves format like: 2026-01-02 21:45:00
    story_data["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 3. Download Image
    image_url = story_data.get("image_url")
    if image_url:
        try:
            # Create a unique filename using the timestamp or a hash
            filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            image_path = os.path.join(image_dir, filename)
            
            # Download and save
            response = requests.get(image_url, stream=True)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                
                # Update JSON to store the local path instead of/alongside the URL
                story_data["local_image_path"] = os.path.join("data", "images", filename)
        except Exception as e:
            print(f"Failed to download image: {e}")

    # 4. Save to JSON
    file_path = os.path.join(data_dir, "stories.json")
    stories = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            try:
                stories = json.load(f)
            except:
                stories = []

    stories.append(story_data)
    with open(file_path, "w") as f:
        json.dump(stories, f, indent=4)

# 6. IMAGE GENERATION - REAL
async def generate_ark_image(image_prompt):
    """Calls Volcengine's Image generating API (Ark)"""

    # Check MOCK_MODE from your config
    if MOCK_MODE:
        # 1. Simulate a realistic delay (e.g., between 3 and 6 seconds) 
        # This allows you to test if your Play Button appears while the image is "loading"
        wait_time = random.uniform(3.0, 6.0)
        await asyncio.sleep(wait_time)

        print(f"✅ MOCK_MODE: Simulation complete after {wait_time:.2f}s")
        return f"https://picsum.photos/seed/{image_prompt[:10]}/800/600"


    url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
    volc_api_key = os.environ.get("VOLC_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {volc_api_key}",
        "Content-Type": "application/json",
    }
    
    request_body = {
        "model": "doubao-seedream-4-5-251128",
        "prompt": f"Super Mario World style pixel art, 16-bit SNES game aesthetic. A beautiful, colorful children's picture book illustration of: {image_prompt}. Friendly characters, vibrant colors, imaginative setting.",
        "size": "2304x1728",
        "num_images": 1
    }

    # Use a longer timeout because image generation is heavy
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=request_body, timeout=60.0)
            data = response.json()
            
            if "data" in data and data["data"]:
                item = data["data"][0]
                return item.get("url") or f"data:image/png;base64,{item.get('b64_image')}"
        except Exception as e:
            print(f"Ark API Error: {e}")
    
    return "https://picsum.photos/800/600?grayscale"

async def real_streaming_generator(topic, level, character_name="Random", specific_words=""):
    full_content = ""

    # Define prompts for different levels
    level_prompts = {
        "easy": """The story should be suitable for kids under 6 years old. 
                Use the 300 most basic English words. 
                Only very simple sentences (e.g., 'Tom is a cat'). 
                Max 7 words per sentence. Max total length: 80 words.""",
        
        "normal": """The story should be suitable for kids aged 6-12. 
                    Use a vocabulary of approximately 800 basic words. 
                    Keep grammar simple (mostly present and past simple). 
                    Max total length: 150 words.""",
        
        "hard": """The story should be suitable for teenagers aged 12-18. 
                Use a vocabulary of up to 5,000 common words. 
                Use varied sentence structures and descriptive language. 
                Max total length: 300 words.""",
        
        "extreme": """The story should be suitable for highly sophisticated readers.
                    Use any English vocabulary, including rare and academic words. 
                    Incorporate complex literary devices and a masterfully poetic style. 
                    Max total length: 500 words."""
    }

    # system_prompt = level_prompts.get(level, level_prompts["normal"])
    system_prompt = """
        You are a creative and imaginative children's story writer, and a certified ESL teacher. You write stories that are engaging, age-appropriate, and easy to understand for young readers learning English as a second language and focus on vocabulary and grammar suitable for their age group. You can adapt your writing style to different age levels, from kindergarten to high school, ensuring that the stories are both educational and entertaining. Try to include more conversations in the story to make it lively and interesting.
        """
 
    user_prompt = f"""
        Write a child story based on the keyword: {topic}. {level_prompts.get(level, level_prompts["normal"])}
        """


    # Add character name if specified
    if character_name and character_name != "random":
        user_prompt += f"\nThe main character is named {character_name}."
    
    # Add specific words if provided
    if specific_words:
        user_prompt += f"\nThe story must include these words: {specific_words}."
    
    system_prompt += "\nFORMAT: ALWAYS start with 'TITLE: ' then a new line, then the story. Output in English only, do not output any Chinese."

    print(f"Generating story on topic: {topic} at level: {level}")
    print(f"System Prompt: {system_prompt}")
    print(f"User Prompt: {user_prompt}")

    # STEP 1: Stream the story text
    response = await client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=1024,
        stream=True
    )

    async for chunk in response:
        if chunk.choices and (content := chunk.choices[0].delta.content):
            full_content += content
            yield content

    # --- AT THIS POINT, THE FRONTEND HAS THE FULL TEXT ---
    # The "Play Audio" button can now be clicked by the user!
    
    # STEP 2: Send a "TEXT_DONE" marker
    # This tells the frontend: "The story is finished, enable the Play button now!"
    yield "||TEXT_COMPLETE||"

    # STEP 3: Background Image Generation (Hidden from user text)
    prompt_res = await client.chat.completions.create(
        model="mimo-v2-flash",
        messages=[
            {"role": "system", "content": "Write a 1-sentence visual prompt in Super Mario World style."},
            {"role": "user", "content": f"Story: {full_content}"}
        ]
    )
    visual_description = prompt_res.choices[0].message.content
    
    image_url = await generate_ark_image(visual_description)
    
    # STEP 4: Send the Image URL with a marker
    yield f"||IMAGE_URL||{image_url}"
    
    save_story({"topic": topic, "text": full_content, "image_url": image_url})

# 6. ROUTES
@app.get("/")
async def read_index():
    # Return index.html from the root folder
    return FileResponse(os.path.join(BASE_DIR, 'index.html'))

@app.post("/generate-story")
async def generate_story(request: StoryRequest):
    return StreamingResponse(real_streaming_generator(request.topic, request.level, request.character_name, request.specific_words), media_type="text/plain")

@app.get("/storybook")
async def get_storybook_page():
    return FileResponse(os.path.join(BASE_DIR, 'storybook.html'))

@app.get("/api/stories")
async def get_all_stories():
    file_path = os.path.join(BASE_DIR, "data", "stories.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    return []

@app.delete("/api/stories/{story_index}")
async def delete_story(story_index: int):
    file_path = os.path.join(BASE_DIR, "data", "stories.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            stories = json.load(f)
        
        # Since frontend reverses the list (newest first), we need to reverse here too
        stories.reverse()
        
        if 0 <= story_index < len(stories):
            story = stories[story_index]
            # Delete image file if it exists
            local_path = story.get("local_image_path")
            if local_path:
                full_path = os.path.join(BASE_DIR, local_path)
                if os.path.exists(full_path):
                    os.remove(full_path)
            # Remove from list
            del stories[story_index]
        
        # Reverse back to original order
        stories.reverse()
        
        with open(file_path, "w") as f:
            json.dump(stories, f, indent=4)
    
    return {"message": "Story deleted successfully"}

# Optional: Serve the data/images folder so the browser can see saved images
app.mount("/data", StaticFiles(directory=os.path.join(BASE_DIR, "data")), name="data")

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    try:
        generator = model(request.text, voice=VOICE_PATH, speed=0.87)
        all_audio = []
        for _, _, audio in generator:
            if audio is not None:
                all_audio.append(audio if isinstance(audio, torch.Tensor) else torch.from_numpy(audio))

        final_audio = torch.cat(all_audio, dim=0)
        buffer = io.BytesIO()
        sf.write(buffer, final_audio.numpy(), 24000, format='WAV')
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")
    
    except Exception as e:
        print(f"TTS Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Serve static files from the root directory (e.g., favicon.ico, apple-touch-icon.png)
app.mount("/", StaticFiles(directory=BASE_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    # Using 8000 as requested
    uvicorn.run(app, host="0.0.0.0", port=8000)