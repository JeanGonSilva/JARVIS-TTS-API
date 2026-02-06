from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from kokoro_onnx import Kokoro
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

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"✅ Arquivo já existe: {filename}")
        return
    
    print(f"⬇️ Baixando {filename}...")
    try:
        # Timeout de 30s para evitar travamentos
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download concluído!")
    except Exception as e:
        print(f"❌ Falha no download de {filename}: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        raise e

def get_model():
    global kokoro_model
    if kokoro_model is not None:
        return kokoro_model
    
    print("🔄 Inicializando IA...")
    try:
        # CORREÇÃO: O nome correto do arquivo na release oficial é .int8.onnx
        # Link: https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0
        model_name = "kokoro-v1.0.int8.onnx"
        model_url = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/{model_name}"
        
        voices_name = "voices-v1.0.bin"
        voices_url = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/{voices_name}"
        
        download_file(model_url, model_name)
        download_file(voices_url, voices_name)
        
        kokoro_model = Kokoro(model_name, voices_name)
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
    return {"status": "online", "model": "v1.0-int8"}

@app.api_route("/speak", methods=["GET", "POST"])
def speak(
    text: str = Query(..., description="Texto"),
    voice: str = Query("pf_dora", description="Voz"),
    lang: str = Query("pt-br", description="Idioma")
):
    try:
        model = get_model()
        
        # Tratamento de idioma
        target_lang = "pt-br"
        if "en" in lang.lower():
            target_lang = "en-us"
        elif "br" in lang.lower():
            target_lang = "pt-br"
            # Se pediu BR mas a voz não é BR, força Dora
            if "pf_" not in voice and "pm_" not in voice:
                voice = "pf_dora"

        print(f"🎤 '{text[:10]}...' | {target_lang} | {voice}")

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
        print(f"❌ Erro: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
