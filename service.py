import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

from config import settings

logger = logging.getLogger("qwen3_tts")


class SynthesizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    voice: str = "default"
    language: str = "Auto"


class TTSService:
    def __init__(self) -> None:
        self.device = self._resolve_device()
        logger.info(f"Initializing Qwen3-TTS on device: {self.device}")

        dtype = torch.bfloat16 if "cpu" not in self.device else torch.float32
        attn_impl = "flash_attention_2" if "cuda" in self.device else "sdpa"

        try:
            self.model = Qwen3TTSModel.from_pretrained(
                settings.MODEL_NAME,
                device_map=self.device,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load with {attn_impl}, falling back to default: {e}")
            self.model = Qwen3TTSModel.from_pretrained(
                settings.MODEL_NAME,
                device_map=self.device,
                dtype=dtype,
            )

        self.voice_cache = self._load_voices()
        logger.info(
            f"Loaded {len(self.voice_cache)} voices: {list(self.voice_cache.keys())}")

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_voices(self) -> dict:
        cache: dict = {}
        voices_path = settings.VOICES_DIR

        if not voices_path.exists():
            logger.warning(f"Voices directory {voices_path} does not exist")
            return cache

        for voice_dir in voices_path.iterdir():
            if not voice_dir.is_dir():
                continue

            manifest_path = voice_dir / "manifest.json"
            if not manifest_path.exists():
                logger.warning(
                    f"Skipping voice {voice_dir.name}: missing manifest.json")
                continue

            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                ref_audio = manifest.get("ref_audio", "reference.wav")
                audio_path = voice_dir / ref_audio
                if not audio_path.exists():
                    logger.warning(
                        f"Skipping voice {voice_dir.name}: missing {ref_audio}")
                    continue

                ref_text = manifest.get("ref_text", "")
                prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio=str(audio_path),
                    ref_text=ref_text,
                    x_vector_only_mode=not ref_text,
                )

                cache[voice_dir.name] = {
                    "prompt": prompt_items,
                    "default_lang": manifest.get("language", "Auto"),
                }
            except Exception as e:
                logger.error(f"Failed to load voice {voice_dir.name}: {e}")

        return cache

    def synthesize(self, text: str, voice: str, language: str) -> bytes:
        voice_data = self.voice_cache[voice]
        target_lang = language if language != "Auto" else voice_data["default_lang"]

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=target_lang,
            voice_clone_prompt=voice_data["prompt"]
        )

        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        return buffer.getvalue()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tts = TTSService()
    yield


app = FastAPI(title="Qwen3-TTS", lifespan=lifespan)


@app.post("/synthesize")
def synthesize(req: SynthesizeRequest) -> Response:
    tts: TTSService = app.state.tts
    if req.voice not in tts.voice_cache:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{req.voice}' not found. Available: {list(tts.voice_cache.keys())}",
        )
    wav_bytes = tts.synthesize(req.text, req.voice, req.language)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/voices")
async def list_voices() -> list[str]:
    return list(app.state.tts.voice_cache.keys())


@app.get("/health")
async def health() -> dict:
    tts: TTSService = app.state.tts
    voices_loaded = len(tts.voice_cache)
    status = "healthy" if voices_loaded > 0 else "degraded"
    return {
        "status": status,
        "device": tts.device,
        "voices_loaded": voices_loaded,
    }
