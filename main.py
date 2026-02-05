from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import io
import os

app = FastAPI(title="Jarvis TTS API")

# CORS (Permitir Base44 e outros)
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
    
    print("🔄 Carregando modelo Kokoro v1.0...")
    try:
        # 1. Baixa o MODELO (Já funcionou no seu log)
        model_path = hf_hub_download(
            repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", 
            filename="onnx/model.onnx"
        )
        
        # 2. CORREÇÃO: Baixa o VOICES.json do mesmo repositório novo
        # O arquivo voices.json está dentro da pasta onnx/ no novo repo
        voices_path = hf_hub_download(
            repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", 
            filename="onnx/voices.json"
        )
        
        kokoro_model = Kokoro(model_path, voices_path)
        print("✅ Modelo carregado com sucesso!")
        return kokoro_model
    except Exception as e:
        print(f"❌ Erro crítico no startup: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
def home():
    return {"status": "online", "endpoints": ["/speak"]}

# CORREÇÃO: Mudamos de /tts para /speak para atender o Base44
# Aceita tanto GET quanto POST para garantir compatibilidade
@app.api_route("/speak", methods=["GET", "POST"])
def speak(
    text: str = Query(..., description="Texto para falar"),
    voice: str = Query("af_bella", description="Voz (af_bella, am_adam, etc)")
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
        print(f"Erro na geração: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
