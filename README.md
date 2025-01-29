# Audio2Chat

Audio2Chat converts multi-speaker audio files into chat format using [AssemblyAI](https://www.assemblyai.com/app) for speaker diarization and optionally Whisper for enhanced transcription.

### Features
- Speaker diarization and transcription using AssemblyAI
- Optional enhanced transcription using Whisper large-v3-turbo
- YouTube video download support
- Word-level timestamp support (can be used for speech-to-text and text-to-speech tasks)
- Structured chat format output

## Installation

```bash
# Install from PyPI
pip install audio2chat

# Or install from source
git clone https://github.com/yourusername/audio2chat.git
cd audio2chat
pip install -e .
```

### Requirements
- Python >=3.8
- FFmpeg (for YouTube downloads)
- CUDA-capable GPU (recommended for Whisper)

Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# MacOS
brew install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

You need to have an Assembly AI account and an API key to use audio2chat. Once you setup an account, you can find the API key on your [dashboard](https://www.assemblyai.com/app).

## Usage

### Command Line

Basic usage:
```bash
# Process local audio file
audio2chat input.wav --api-key YOUR_ASSEMBLYAI_KEY --output output_dir

# Process YouTube video
audio2chat "https://youtube.com/watch?v=xxxxx" --api-key YOUR_ASSEMBLYAI_KEY --output output_dir
```

All options:
```bash
audio2chat --help

required arguments:
  input                   Input audio file path or YouTube URL
  --api-key API_KEY      AssemblyAI API key

output settings:
  --output OUTPUT        Output directory for audio and chat data (default: output)
  --download-format {mp3,wav}
                        Audio format for YouTube downloads (default: wav)

transcription settings:
  --language LANGUAGE    Language code for transcription (default: en)
  --num-speakers NUM     Expected number of speakers (default: auto-detect)
  --use-whisper         Use Whisper for enhanced transcription (default: False)

chat generation settings:
  --min-segment-confidence CONF
                        Minimum confidence score to include segment (default: 0.5)
  --merge-threshold THRESH
                        Time threshold to merge adjacent utterances (default: 1.0)
  --min-duration DUR    Minimum duration for a chat segment (default: 0.5)
  --include-metadata    Include additional metadata in output (default: True)
  --include-word-timestamps
                        Include word-level timing information (default: False)

vocabulary settings:
  --word-boost [WORDS ...]
                        List of words to boost recognition for

other:
  --verbose, -v         Enable verbose logging
```

### Python API

```python
from audio2chat.pipeline import AudioChatPipeline
from audio2chat.youtube_downloader import download_audio

# For YouTube videos
audio_path = download_audio(
    "https://youtube.com/watch?v=xxxxx",
    output_dir="downloads",
    audio_format="wav"
)

# Initialize pipeline
pipeline = AudioChatPipeline(
    api_key="YOUR_ASSEMBLYAI_KEY",
    language="en",
    num_speakers=2,  # or None for auto-detect
    use_whisper=True,  # enable Whisper for better transcription
    include_word_timestamps=True
)

# Process file
chat_data = pipeline.process_file(audio_path, "output/chat.json")
```

### Output Format

```json
{
    "messages": [
        {
            "speaker": "A",
            "text": "Hello there!",
            "start": 0,
            "end": 1500,
            "words": [
                {
                    "text": "Hello",
                    "start": 0,
                    "end": 750,
                    "confidence": 0.98
                },
                {
                    "text": "there",
                    "start": 750,
                    "end": 1500,
                    "confidence": 0.95
                }
            ]
        }
    ],
    "metadata": {
        "num_speakers": 2,
        "speakers": ["A", "B"],
        "transcription": "whisper+assemblyai"
    }
}
```

## Development

Run tests:
```bash
# Set up environment
export ASSEMBLYAI_API_KEY=your_key_here

# Add test audio file
cp your_test_audio.wav tests/test_data/input.wav

# Run tests
pytest tests/test_pipeline.py tests/test_chat_builder.py  # without Whisper
pytest tests/  # all tests including Whisper
```

## License
This project is licensed under the [MIT license](https://github.com/neuralwork/audio2chat/blob/main/LICENSE).

From [neuralwork](https://neuralwork.ai/) with :heart:
