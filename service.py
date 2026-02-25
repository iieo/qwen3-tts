import os
import torch
import json
import logging
import bentoml
from pathlib import Path
import soundfile as sf
import io

# Optional build-time import for model
with bentoml.importing():
    from qwen_tts import Qwen3TTSModel

from config import settings

logger = logging.getLogger("bentoml.qwen3_tts")


@bentoml.service(
    name="qwen3-tts",
    resources={"gpu": settings.NUM_GPUS} if settings.NUM_GPUS > 0 else {
        "cpu": 2},
    workers=settings.NUM_WORKERS,
    traffic={"timeout": 300},
)
class TTSService:
    def __init__(self):
        self.device = self._resolve_device()
        logger.info(f"Initializing Qwen3-TTS on device: {self.device}")

        # Load model
        dtype = torch.bfloat16 if "cpu" not in self.device else torch.float32

        # FlashAttention 2 is only supported on CUDA with specific hardware
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

    def _resolve_device(self):
        """Resolves device based on environment and worker index."""
        if torch.cuda.is_available():
            # BentoML worker_index is 1-based, we want 0-based for GPU ID
            worker_idx = bentoml.server_context.worker_index - 1
            if worker_idx < 0:
                worker_idx = 0

            # If multiple GPUs, distribute workers.
            # If 1 GPU, map all workers to it (or let torch handle if device_map="auto")
            gpu_id = worker_idx % torch.cuda.device_count()
            return f"cuda:{gpu_id}"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_voices(self):
        """Scans the voices directory and caches prompt embeddings."""
        cache = {}
        voices_path = settings.VOICES_DIR

        if not voices_path.exists():
            logger.warning(f"Voices directory {voices_path} does not exist")
            return cache

        for voice_dir in voices_path.iterdir():
            if not voice_dir.is_dir():
                continue

            manifest_path = voice_dir / "manifest.json"
            audio_path = voice_dir / "reference.wav"

            if not manifest_path.exists() or not audio_path.exists():
                logger.warning(
                    f"Skipping voice {voice_dir.name}: missing manifest.json or reference.wav")
                continue

            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                ref_text = manifest.get("ref_text", "")

                # Pre-compute voice embedding
                prompt_items = self.model.create_voice_clone_prompt(
                    ref_audio=str(audio_path),
                    ref_text=ref_text,
                    x_vector_only_mode=False if ref_text else True
                )

                cache[voice_dir.name] = {
                    "prompt": prompt_items,
                    "default_lang": manifest.get("language", "Auto")
                }
            except Exception as e:
                logger.error(f"Failed to load voice {voice_dir.name}: {e}")

        return cache

    @bentoml.api
    def synthesize(self, text: str, voice: str = "default", language: str = "Auto") -> bytes:
        """Synthesizes speech using a cloned voice."""
        if voice not in self.voice_cache:
            raise ValueError(
                f"Voice '{voice}' not found. Available: {list(self.voice_cache.keys())}")

        voice_data = self.voice_cache[voice]
        target_lang = language if language != "Auto" else voice_data["default_lang"]

        # Generate audio
        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=target_lang,
            voice_clone_prompt=voice_data["prompt"]
        )

        # Convert numpy array to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format='WAV')
        return buffer.getvalue()

    @bentoml.api(route="/voices")
    def list_voices(self) -> list[str]:
        """Returns the list of cached voices."""
        return list(self.voice_cache.keys())

    @bentoml.get("/health")
    def health(self) -> dict:
        """Simple health check."""
        return {
            "status": "healthy",
            "device": self.device,
            "voices_loaded": len(self.voice_cache)
        }
