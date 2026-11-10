import os
from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient, AudioConfig 

load_dotenv()

def get_api_key():
    api_key = os.getenv("Gnani_apkey")

    if not api_key:
        raise ValueError(
            "Gnani_Api_key is not configured."
        )

    return api_key

# Speech to text

def transcribe_audio(file):
    if not file:
        raise ValueError(
            "Audio filename is required."
        )
    audio_bytes = file.read()
    if not audio_bytes:
        raise ValueError(
            "Uploaded audio  is empty."
        )
    try:
        client = GnaniSTTClient(api_key = get_api_key())

        result = client.transcribe_bytes(
                audio_bytes,
                filename=file.name,
                language_code="indic-auto",
                model="prisma-v3.0",
                format="codemix"
            )
        
        return result["transcript"]
    
    except Exception as e :
        raise RuntimeError(
            f"Speech recognition failed: {str(e)}"
        )

# Text to Speech

def synthesize_speech(text):

    if not text.strip():
        raise ValueError(
            "Text cannot be empty"
        )
    try:
        client = GnaniTTSClient(api_key=get_api_key())

        audio =  client.synthesize(
                text,
                voice="Avinash",
                model="timbre-v3.0",
                audio_config=AudioConfig(
                    container="wav",
                    encoding="linear_pcm",
                    sample_rate=22050
                )
        )
        

        return audio

    except Exception as e:
        raise RuntimeError(
            f"Speech generation failed {str(e)}"
        )