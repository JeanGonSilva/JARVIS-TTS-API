from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uuid
import io
from pydub import AudioSegment
import os

app = FastAPI(title="Jarvis TTS API")

# CORS liberado para Base44
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos do Piper e modelo
PIPER_PATH = "./piper"  # binário Piper no projeto
MODEL_PATH = "./models/pt_BR-edresson-medium.onnx"

@app.post("/tts")
def tts(text: str = Query(..., description="Texto para gerar voz")):
    """
    Recebe texto e retorna áudio MP3 gerado pelo Piper.
    """

    # Arquivo temporário WAV
    wav_path = f"/tmp/{uuid.uuid4()}.wav"

    # Executa Piper para gerar WAV
    try:
        subprocess.run(
            [PIPER_PATH, "--model", MODEL_PATH, "--output_file", wav_path],
            input=text,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        return {"error": "Falha ao gerar áudio", "details": str(e)}

    # Converte WAV → MP3 em memória
    audio = AudioSegment.from_wav(wav_path)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3")
    mp3_io.seek(0)

    # Remove WAV temporário
    os.remove(wav_path)

    # Retorna MP3 como streaming
    return StreamingResponse(mp3_io, media_type="audio/mpeg")
