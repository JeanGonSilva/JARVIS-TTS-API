from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import requests
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
    
    print("🔄 Iniciando carregamento do modelo...")
    try:
        # 1. Baixa o MODELO ONNX (Cache do HuggingFace)
        print("   - Verificando Model ONNX...")
        model_path = hf_hub_download(
            repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", 
            filename="onnx/model.onnx"
        )
        
        # 2. CORREÇÃO: Baixa o novo formato VOICES.BIN (Do GitHub Releases)
        # O voices.json morreu. O novo padrão é .bin
        voices_file = "voices.bin"
        if not os.path.exists(voices_file):
            print("   - Baixando voices.bin (novo formato)...")
            # Link direto da Release Oficial v1.0
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(voices_file, "wb") as f:
                    f.write(response.content)
            else:
                raise Exception(f"Falha ao baixar voices.bin: Status {response.status_code}")
        
        voices_path = os.path.abspath(voices_file)
        
        # 3. Inicializa (A lib kokoro-onnx aceita o .bin automaticamente)
        kokoro_model = Kokoro(model_path, voices_path)
        print("✅ Modelo carregado com sucesso!")
        return kokoro_model
    except Exception as e:
        print(f"❌ Erro crítico no startup: {e}")
        # Tenta apagar o arquivo se estiver corrompido para baixar de novo no próximo boot
        if os.path.exists("voices.bin"):
            os.remove("voices.bin")
        raise e

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
def home():
    return {"status": "online", "endpoints": ["/speak"]}

@app.api_route("/speak", methods=["GET", "POST"])
def speak(
    text: str = Query(..., description="Texto para falar"),
    voice: str = Query("af_bella", description="Voz (af_bella, am_adam, etc)")
):
    try:
        model = get_model()
        
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
