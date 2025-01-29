from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="audio2chat",
    version="0.1.0",
    author="Alara Dirik",
    description="Generate chat data from multi-speaker audio files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuralwork/audio2chat",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=2.1.1",  
        "transformers>=4.37.0",
        "accelerate>=0.27.0",
        "datasets[audio]>=2.16.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "assemblyai",
        "ffmpeg-python>=0.2.0",
        "yt-dlp>=2023.12.30",
    ],
    extras_require={
        'speed': [
            "flash-attn>=2.5.0",
            "safetensors>=0.4.1",
        ],
        'dev': [
            "pytest>=6.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "audio2chat=audio2chat.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="audio, transcription, whisper, diarization, speech-to-text, chat",
)