# Character Voice Setup (Text-to-Speech)

## Recommended: Edge TTS

Zero setup, free, runs without a GPU. Microsoft provides 300+ voices with real accent variation.

### Install

```bash
pip install edge-tts
```

### Voice Map per Agent

| Agent | Voice ID | Accent |
|-------|----------|--------|
| Villager | `en-GB-ThomasNeural` | Welsh-ish British |
| Mayor | `fr-FR-HenriNeural` | French |
| Judge | `en-GB-RyanNeural` | Neutral British |
| Captain | `en-GB-ThomasNeural` | British (gruff) |
| TownCrier | `en-GB-RyanNeural` | Neutral British |

### List All Available Voices

```bash
edge-tts --list-voices | grep "en-GB\|fr-FR\|en-IE\|cy-GB"
```

### Generate Audio from Text

```bash
edge-tts --voice en-GB-ThomasNeural --text "Oi! What about the bandits in them woods?" --write-media villager.mp3
```

### Python Usage

```python
import edge_tts
import asyncio

VOICE_MAP = {
    "Villager": "en-GB-ThomasNeural",
    "Mayor": "fr-FR-HenriNeural",
    "Judge": "en-GB-RyanNeural",
    "Captain": "en-GB-ThomasNeural",
}

async def speak(role: str, text: str, output_path: str) -> None:
    voice = VOICE_MAP.get(role, "en-GB-RyanNeural")
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

# Example
asyncio.run(speak("Mayor", "Mes amis, we must act now!", "mayor_line.mp3"))
```

### Integrate into Agent.respond()

To auto-generate audio after each response, add to `Agent.respond()`:

```python
async def respond(self, text: str) -> None:
    print(f"\n{self.role} responding: {text}")
    await redis_client.xadd(
        self.stream_name,
        {"role": self.role, "text": text}
    )

    # Generate audio
    output_path = f"audio/{self.role}_{self.replies_sent}.mp3"
    await speak(self.role, text, output_path)

    await asyncio.sleep(self.reply_cooldown_seconds)
```

---

## Alternative: Coqui XTTS v2 (Voice Cloning)

For truly distinct character voices cloned from audio samples. Requires a GPU.

### Install

```bash
pip install TTS
```

### Usage

```python
from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Provide a ~6 second .wav sample of the desired voice
tts.tts_to_file(
    "Good people of the town!",
    speaker_wav="samples/welsh_voice.wav",
    language="en",
    file_path="villager.wav"
)
```

You would need one short `.wav` sample per character to define their voice.
