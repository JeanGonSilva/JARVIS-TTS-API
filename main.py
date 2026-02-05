from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import io
import os

app = FastAPI(title="Jarvis TTS API (Kokoro v1.0)")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

kokoro_model = None

def get_model():
    global kokoro_model
    if kokoro_model is not None:
        return kokoro_model
    
    print("🔄 Baixando modelo Kokoro v1.0...")
    try:
        # ATUALIZAÇÃO: Usando a versão v1.0 do repositório ONNX Community
        # Baixa o modelo ONNX (v1.0)
        model_path = hf_hub_download(
            repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", 
            filename="onnx/model.onnx"
        )
        
        # Baixa o arquivo de vozes (voices.json ainda é compatível e mais fácil de achar)
        voices_path = hf_hub_download(
            repo_id="hexgrad/Kokoro-82M", 
            filename="voices.json"
        )
        
        kokoro_model = Kokoro(model_path, voices_path)
        print("✅ Modelo carregado com sucesso!")
        return kokoro_model
    except Exception as e:
        print(f"❌ Erro crítico ao carregar modelo: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
def home():
    return {"status": "online", "version": "v1.0-ONNX"}

@app.get("/tts")
def tts(
    text: str = Query(..., description="Texto para falar"),
    voice: str = Query("af_bella", description="ID da voz (ex: af_bella, am_adam)")
):
    try:
        model = get_model()
        
        # Gera áudio
        audio, sample_rate = model.create(
            text,
            voice=voice,
            speed=1.0,
            lang="en-us"
        )
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        print(f"Erro na geração: {str(e)}") # Log no console do Render
        return JSONResponse(status_code=500, content={"error": str(e)})
