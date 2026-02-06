from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import requests
import io
import os
import gc

app = FastAPI(title="Jarvis TTS API (Multi-Language)")

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
    
    print("🔄 Carregando modelo LIGHT (Multi-Idiomas)...")
    try:
        # 1. Baixa o Modelo Quantizado (~87MB)
        model_path = hf_hub_download(
            repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", 
            filename="onnx/model_quantized.onnx" 
        )
        
        # 2. Baixa o arquivo de Vozes (voices.bin)
        voices_file = "voices.bin"
        if not os.path.exists(voices_file):
            print("   - Baixando voices.bin...")
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            response = requests.get(url, allow_redirects=True)
            with open(voices_file, "wb") as f:
                f.write(response.content)
        
        voices_path = os.path.abspath(voices_file)
        
        # 3. Inicializa
        gc.collect()
        kokoro_model = Kokoro(model_path, voices_path)
        print("✅ Modelo carregado!")
        return kokoro_model
    except Exception as e:
        print(f"❌ Erro crítico: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
def home():
    return {
        "status": "online", 
        "info": "Use /speak?text=Ola&lang=pt-br&voice=pf_dora",
        "voices_br": ["pf_dora (Mulher)", "pm_alex (Homem)", "pm_santa (Papai Noel)"]
    }

@app.api_route("/speak", methods=["GET", "POST"])
def speak(
    text: str = Query(..., description="Texto para falar"),
    voice: str = Query("pf_dora", description="Voz: pf_dora (BR), pm_alex (BR), af_bella (EN)"),
    lang: str = Query("pt-br", description="Idioma: pt-br ou en-us")
):
    try:
        model = get_model()
        
        # Mapeamento de idiomas para o código do Kokoro
        # 'p' = Português, 'a' = American English, 'b' = British English
        lang_code = 'p' if lang.lower() in ['pt', 'pt-br', 'br'] else 'a'
        
        # Se o usuário pedir inglês explicitamente
        if lang.lower() in ['en', 'en-us']:
            lang_code = 'a'
        
        audio, sample_rate = model.create(
            text,
            voice=voice,
            speed=1.0,
            lang=lang_code
        )
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        del audio
        gc.collect()
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
