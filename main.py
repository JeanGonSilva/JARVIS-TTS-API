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

app = FastAPI(title="Jarvis TTS API (Debug Mode)")

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
        
        # Tenta inicializar. O espeak-ng costuma ser o vilão aqui.
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
    return {"status": "online", "debug": "true"}

@app.api_route("/speak", methods=["GET", "POST"])
def speak(
    text: str = Query(..., description="Texto"),
    voice: str = Query("pf_dora", description="Voz"),
    lang: str = Query("pt-br", description="Idioma")
):
    try:
        print(f"📩 Recebido: '{text}' | Voz: {voice} | Lang: {lang}")
        model = get_model()
        
        # Mapeamento simplificado para teste
        lang_code = 'p' if 'pt' in lang.lower() else 'a'
        if lang_code == 'p' and 'pf_' not in voice and 'pm_' not in voice:
             # Se pedir pt-br mas usar voz americana, força voz BR padrão
             voice = "pf_dora"

        print(f"⚙️ Gerando: lang_code='{lang_code}', voice='{voice}'")

        audio, sample_rate = model.create(
            text,
            voice=voice,
            speed=1.0,
            lang=lang_code
        )
        
        print(f"✅ Áudio gerado! Tamanho: {len(audio)}")

        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        del audio
        gc.collect()
        
        return StreamingResponse(buffer, media_type="audio/wav")

    except Exception as e:
        # ISSO VAI MOSTRAR O ERRO REAL NO LOG DO RENDER
        error_msg = f"❌ ERRO FATAL: {str(e)}"
        print(error_msg) 
        return JSONResponse(status_code=500, content={"error": str(e), "details": "Verifique os logs do Render"})
