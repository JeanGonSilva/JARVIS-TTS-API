import asyncio
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
try:
    import kokoro
except ImportError:
    kokoro = None

app = FastAPI(
    title="JARVIS TTS API",
    version="2.0.0",
    description="FREE JARVIS-style Text-to-Speech API - Natural human-like voice"
)

# Available voices with descriptions
VOICES = {
    "bella": "Female voice - warm and friendly",
    "adam": "Male voice - professional and clear",
    "sarah": "Female voice - soft and natural",
    "josh": "Male voice - energetic and dynamic",
    "male1": "Male voice - British accent",
    "female1": "Female voice - British accent",
}

@app.get("/")
async def root():
    return {
        "service": "JARVIS TTS API",
        "version": "2.0.0",
        "description": "Free JARVIS-style Text-to-Speech powered by Kokoro TTS",
        "endpoints": {
            "speak": "/speak?text=YOUR_TEXT&voice=bella&lang=en",
            "voices": "/voices",
            "health": "/health"
        },
        "features": [
            "Natural human-like voice",
            "No API keys required",
            "Unlimited usage",
            "Multi-language support",
            "Completely FREE"
        ]
    }

@app.get("/voices")
async def list_voices():
    return {
        "available_voices": VOICES,
        "default_voice": "bella",
        "example": "/speak?text=Hello%20World&voice=bella&lang=en"
    }

@app.get("/speak")
async def text_to_speech(text: str, voice: str = "bella", lang: str = "en"):
    """
    Convert text to speech with JARVIS-like voice
    
    Parameters:
    - text: The text to speak (required)
    - voice: Voice to use (bella, adam, sarah, josh, male1, female1)
    - lang: Language (en=English, hi=Hindi, multi=Multilingual)
    
    Returns: MP3 audio stream
    """
    
    if not text or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text parameter is required")
    
    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not found. Available: {list(VOICES.keys())}"
        )
    
    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 characters)")
    
    try:
        # Use Kokoro TTS if available, otherwise use fallback
        if kokoro:
            audio_buffer = io.BytesIO()
            
            # Generate speech using Kokoro
            samples = kokoro.generate(
                text=text,
                voice=voice,
                speed=1.0
            )
            
            # Convert samples to MP3 format
            import wave
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(24000)
                wav_file.writeframes(samples)
            
            wav_buffer.seek(0)
            audio_buffer = wav_buffer
        else:
            # Fallback: return error with helpful message
            raise HTTPException(
                status_code=503,
                detail="Kokoro TTS not installed. Install with: pip install -U git+https://github.com/hexgrad/kokoro"
            )
        
        audio_buffer.seek(0)
        return StreamingResponse(
            iter([audio_buffer.getvalue()]),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=speech.wav"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "JARVIS TTS API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
