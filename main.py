import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

try:
    import kokoro
except ImportError:
    kokoro = None

import numpy as np
from pydub import AudioSegment  # para converter WAV para MP3

app = FastAPI(
    title="JARVIS TTS API",
    version="2.0.0",
    description="FREE JARVIS-style Text-to-Speech API - Natural human-like voice"
)

VOICES = {
    "bella": "Female voice - warm and friendly",
    "adam": "Male voice - professional and clear",
    "sarah": "Female voice - soft and natural",
    "josh": "Male voice - energetic and dynamic",
    "male1": "Male voice - British accent",
    "female1": "Female voice - British accent",
}

@app.get("/speak")
async def text_to_speech(text: str, voice: str = "bella", lang: str = "en"):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text parameter is required")

    if voice not in VOICES:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not found. Available: {list(VOICES.keys())}"
        )

    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long (max 2000 characters)")

    if not kokoro:
        raise HTTPException(
            status_code=503,
            detail="Kokoro TTS not installed. Install with: pip install -U git+https://github.com/hexgrad/kokoro"
        )

    try:
        # Gerar fala com Kokoro
        samples: np.ndarray = kokoro.generate(
            text=text,
            voice=voice,
            speed=1.0
        )

        # Converter samples para WAV usando AudioSegment
        audio_buffer = io.BytesIO()
        # AudioSegment precisa de int16
        audio_segment = AudioSegment(
            samples.tobytes(), 
            frame_rate=24000,
            sample_width=2, 
            channels=1
        )
        # Exportar para MP3
        audio_segment.export(audio_buffer, format="mp3")
        audio_buffer.seek(0)

        return StreamingResponse(
            iter([audio_buffer.getvalue()]),
            media_type="audio/mpeg",
            headers={"Content-Disposition": f"attachment; filename=speech.mp3"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")
