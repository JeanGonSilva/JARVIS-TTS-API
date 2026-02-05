from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import io
import os

app = FastAPI(title="Jarvis TTS API (Kokoro)")

# CORS (Permitir Base44 e outros)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variável global para o modelo
kokoro_model = None

def get_model():
    """Carrega o modelo apenas uma vez (Singleton) e baixa se necessário"""
    global kokoro_model
    if kokoro_model is not None:
        return kokoro_model
    
    print("🔄 Baixando/Carregando modelo Kokoro...")
    try:
        # Baixa os arquivos necessários do HuggingFace automaticamente
        model_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="kokoro-v0_19.onnx")
        voices_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="voices.json")
        
        kokoro_model = Kokoro(model_path, voices_path)
        print("✅ Modelo carregado com sucesso!")
        return kokoro_model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    # Pré-carrega o modelo ao iniciar o servidor
    get_model()

@app.get("/")
def home():
    return {"status": "online", "engine": "Kokoro ONNX"}

@app.get("/tts")
def tts(
    text: str = Query(..., description="Texto para falar"),
    voice: str = Query("af_bella", description="ID da voz (ex: af_bella, af_sarah, am_adam)")
):
    try:
        model = get_model()
        
        # Gera o áudio (retorna raw audio data e sample rate)
        # O Kokoro ONNX é muito rápido
        audio, sample_rate = model.create(
            text,
            voice=voice,
            speed=1.0,
            lang="en-us" # Kokoro é focado em inglês, mas aceita pt com sotaque se usar phonemes (avançado)
        )
        
        # Converte para arquivo em memória (WAV/FLAC)
        # WAV é mais seguro que MP3 aqui pois não exige ffmpeg instalado no linux
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Para rodar localmente: uvicorn main:app --reload
