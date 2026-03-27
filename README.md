# qwen3-tts

TTS service based on [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) with voice cloning support.

## Setup

```bash
cp .env.example .env
uv sync
```

## Run

```bash
uv run python main.py
```

Or with Docker:

```bash
docker build -t qwen3-tts .
docker run -p 3000:3000 qwen3-tts
```

## API

```bash
# Synthesize speech
curl -X POST http://localhost:3000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "default", "language": "English"}' \
  --output output.wav

# List voices
curl http://localhost:3000/voices

# Health check
curl http://localhost:3000/health
```

## Adding a Voice

1. Create a directory under `voices/`:

```
voices/
└── my-voice/
    ├── manifest.json
    └── reference.wav   # or .mp3
```

2. Create `manifest.json`:

```json
{
  "ref_audio": "reference.wav",
  "ref_text": "Transcription of the reference audio.",
  "language": "English"
}
```

- `ref_audio` — audio file in the same directory (WAV or MP3, a few seconds is enough)
- `ref_text` — exact transcription of the audio; leave empty `""` to use x-vector mode (speaker embedding only)
- `language` — default language for this voice (`English`, `German`, `Auto`, …)

3. Restart the service. The voice is available immediately under its directory name.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID |
| `VOICES_DIR` | `./voices` | Path to voices directory |
| `PORT` | `3000` | Server port |
