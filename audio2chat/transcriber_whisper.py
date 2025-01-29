from typing import Dict
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperTranscriber:
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        batch_size: int = 8,
        chunk_length_s: int = 30,
        sampling_rate: int = 16000,
        use_flash_attention: bool = False,
        language: str = "en"
    ):
        """
        Initialize the transcriber with Transformers Whisper implementation
        
        Args:
            model_id: Whisper model identifier
            batch_size: Batch size for parallel processing
            chunk_length_s: Length of audio chunks in seconds
            sampling_rate: Target sampling rate for the audio
            use_flash_attention: Whether to use Flash Attention 2 if available
            language: Language code for transcription
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.sampling_rate = sampling_rate
        self.language = language
        
        # Initialize model with optimized settings
        kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
        
        if use_flash_attention and torch.cuda.is_available():
            try:
                import flash_attn
                kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                print("Flash Attention 2 not available. Using default attention implementation.")
        
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs)
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # create pipeline with optimized settings
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps="word"
        )

    def transcribe_utterance(self, audio_path: str, start: float, end: float) -> Dict:
        """
        Transcribe a specific utterance from an audio file
        
        Args:
            audio_path: Path to audio file
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Dict containing transcribed text and word timestamps
        """
        import librosa
        
        # add small padding to avoid cutting off words
        pad = 0.1  # 100ms padding
        start = max(0, start - pad)
        
        # load audio segment
        try:
            audio, sr = librosa.load(
                audio_path,
                sr=self.sampling_rate,
                offset=start,
                duration=(end - start) + 2*pad  # Add padding to both ends
            )
        except Exception as e:
            raise Exception(f"Error loading audio segment: {e}")
        
        # get transcription with word-level timestamps
        try:
            result = self.pipe(
                audio,
                return_timestamps="word",
                generate_kwargs={
                    "language": self.language,
                    "task": "transcribe",
                    "num_beams": 5
                }
            )
            print(result)
        except Exception as e:
            raise Exception(f"Error during transcription: {e}")
        
        # adjust timestamps to account for segment offset
        words = []
        for chunk in result.get("chunks", []):
            # only include words within the original segment bounds
            word_start = chunk["timestamp"][0] + start
            word_end = chunk["timestamp"][1] + start
            
            if word_start >= start and word_end <= end:
                words.append({
                    "text": chunk["text"].strip(),
                    "start": word_start,
                    "end": word_end
                })

        return {
            "text": result["text"].strip(),
            "words": words
        }

    def transcribe_file(self, audio_path: str) -> Dict:
        """
        Transcribe an entire audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict containing transcribed text and word timestamps
        """
        try:
            result = self.pipe(
                audio_path,
                return_timestamps="word",
                generate_kwargs={
                    "language": self.language,
                    "task": "transcribe",
                    "num_beams": 5
                }
            )
            return result
        except Exception as e:
            raise Exception(f"Error transcribing file: {e}")