from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
from huggingface_hub import hf_hub_download
import soundfile as sf
import requests
import io
import os
import gc

app = FastAPI(title="Jarvis TTS API")

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
    
    print("🔄 Carregando modelo...")
    try:
        model_path = hf_hub_download(
            repo_id="onnx-community/Kokoro-82M-v1.0-ONNX", 
            filename="onnx/model_quantized.onnx" 
        )
        
        voices_file = "voices.bin"
        if not os.path.exists(voices_file):
            print("   - Baixando voices.bin...")
            url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
            response = requests.get(url, allow_redirects=True)
            with open(voices_file, "wb") as f:
                f.write(response.content)
        
        voices_path = os.path.abspath(voices_file)
        
        kokoro_model = Kokoro(model_path, voices_path)
        print("✅ Modelo carregado!")
        return kokoro_model
    except Exception as e:
        print(f"❌ Erro crítico no startup: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
def home():
    return {"status": "online"}

@app.api_route("/speak", methods=["GET", "POST"])
def speak(
    text: str = Query(..., description="Texto"),
    voice: str = Query("pf_dora", description="Voz (pf_dora, pm_alex, af_bella)"),
    lang: str = Query("pt-br", description="Idioma (pt-br, en-us)")
):
    try:
        model = get_model()
        
        # CORREÇÃO: Usar códigos ISO reais (pt-br, en-us)
        # A biblioteca kokoro-onnx cuida da conversão interna.
        target_lang = "pt-br"
        
        if "en" in lang.lower():
            target_lang = "en-us"
        elif "br" in lang.lower() or "pt" in lang.lower():
            target_lang = "pt-br"
            
            # Garante voz BR se o usuário pedir PT-BR mas usar voz gringa
            if "pf_" not in voice and "pm_" not in voice:
                voice = "pf_dora"

        print(f"🎤 Falando: '{text[:20]}...' | Lang: {target_lang} | Voz: {voice}")

        audio, sample_rate = model.create(
            text,
            voice=voice,
            speed=1.0,
            lang=target_lang 
        )
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        del audio
        gc.collect()
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        print(f"❌ Erro: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
