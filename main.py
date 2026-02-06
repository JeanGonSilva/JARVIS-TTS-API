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
    """Função auxiliar para baixar arquivos grandes com segurança"""
    if os.path.exists(filename):
        print(f"✅ Arquivo encontrado: {filename}")
        return
    
    print(f"⬇️ Baixando {filename} de {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Download concluído: {filename}")
    except Exception as e:
        print(f"❌ Falha ao baixar {filename}: {e}")
        # Remove arquivo corrompido se falhar
        if os.path.exists(filename):
            os.remove(filename)
        raise e

def get_model():
    global kokoro_model
    if kokoro_model is not None:
        return kokoro_model
    
    print("🔄 Inicializando sistema de IA...")
    try:
        # 1. Baixar MODELO QUANTIZADO OFICIAL (Release v1.0)
        # Fonte: Releases do repositório oficial kokoro-onnx
        model_filename = "kokoro-v1.0_quant.onnx"
        model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0_quant.onnx"
        download_file(model_url, model_filename)
        
        # 2. Baixar VOZES (Release v1.0)
        voices_filename = "voices.bin"
        voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        download_file(voices_url, voices_filename)
        
        # Caminhos absolutos
        model_path = os.path.abspath(model_filename)
        voices_path = os.path.abspath(voices_filename)
        
        # Inicializa
        kokoro_model = Kokoro(model_path, voices_path)
        print("✅ Modelo carregado na memória!")
        return kokoro_model
    except Exception as e:
        print(f"❌ Erro crítico no startup: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    get_model()

@app.get("/")
def home():
    return {"status": "online", "model": "kokoro-v1.0_quant"}

@app.api_route("/speak", methods=["GET", "POST"])
def speak(
    text: str = Query(..., description="Texto"),
    voice: str = Query("pf_dora", description="Voz (pf_dora, pm_alex)"),
    lang: str = Query("pt-br", description="Idioma (pt-br, en-us)")
):
    try:
        model = get_model()
        
        # Lógica de idioma
        target_lang = "pt-br"
        if "en" in lang.lower():
            target_lang = "en-us"
        elif "br" in lang.lower() or "pt" in lang.lower():
            target_lang = "pt-br"
            # Fallback para voz BR se necessário
            if "pf_" not in voice and "pm_" not in voice:
                voice = "pf_dora"

        print(f"🎤 Processando: '{text[:15]}...' | Lang: {target_lang} | Voz: {voice}")

        audio, sample_rate = model.create(
            text,
            voice=voice,
            speed=1.0,
            lang=target_lang 
        )
        
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        # Limpeza de memória
        del audio
        gc.collect()
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        print(f"❌ Erro na geração: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
