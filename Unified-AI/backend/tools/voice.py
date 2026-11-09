import os

from io import BytesIO
from dotenv import load_dotenv
from fastapi import File, HTTPException, UploadFile
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient 

load_dotenv()

class SpeechRequest(BaseModel):
    text : str

def get_api_key():
    api_key = os.getenv("Gnani_apkey")

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="Gnani_Api_key is not configured."
        )

    return api_key

# Speech to text

async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Audio filename is required."
        )
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded audio  is empty."
        )
    try:
        client = GnaniSTTClient(api_key = get_api_key())

        result = await run_in_threadpool(
            lambda: client.transcribe(
                BytesIO(audio_bytes),
                filename= file.filename,
                language_code="indic-auto",
                model="prisma-v3.0",
                format="codemix"
            )
        )
        return {
            "transcript": result["transcript"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Speech recognition failed: {str(e)}"
        )

# Text to Speech

async def synthesize_speech(request: SpeechRequest):

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )
    try:
        client = GnaniTTSClient(api_key=get_api_key())

        audio = await run_in_threadpool(
            lambda: client.synthesize(
                request.text,
                voice="pranav",
                model="timbre-v2.0"
            )
        )

        return Response(
            content=audio,
            media_type="audio/wav",
            headers={
                "Content-Disposition": ' inline; filename="response.wav"'
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail= f"Speech generation failed: {str(e)}"
        )