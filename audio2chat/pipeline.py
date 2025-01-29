import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import assemblyai as aai
from .chat_builder import ChatBuilder
from .transcriber_whisper import WhisperTranscriber


class AudioChatPipeline:
    def __init__(
        self,
        api_key: str,
        language: str = 'en',
        num_speakers: Optional[int] = None,
        word_boost: Optional[List[str]] = None,
        min_segment_confidence: float = 0.5,
        merge_threshold: float = 1.0,
        min_duration: float = 0.5,
        include_metadata: bool = True,
        include_word_timestamps: bool = True,
        use_whisper: bool = True
    ):
        """
        Initialize the audio to chat pipeline
        
        Args:
            api_key: AssemblyAI API key
            language: Language code for transcription
            num_speakers: Expected number of speakers (None for auto-detect)
            word_boost: List of words to boost recognition for
            min_segment_confidence: Minimum confidence score to include segment
            merge_threshold: Time threshold to merge adjacent utterances
            min_duration: Minimum duration for a chat segment
            include_metadata: Whether to include additional metadata
            include_word_timestamps: Whether to include word-level timing
            use_whisper: Whether to use Whisper for enhanced transcription
        """
        # set up AssemblyAI for diarization
        aai.settings.api_key = api_key
        self.transcription_config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            speaker_labels=True,
            language_code=language,
            speakers_expected=num_speakers,
            word_boost=word_boost
        )
        
        self.assemblyai = aai.Transcriber(config=self.transcription_config)
        self.whisper = WhisperTranscriber(language=language) if use_whisper else None
        self.chat_builder = ChatBuilder(
            min_segment_confidence=min_segment_confidence,
            merge_threshold=merge_threshold,
            min_duration=min_duration,
            include_metadata=include_metadata,
            include_word_timestamps=include_word_timestamps
        )
        self.logger = logging.getLogger(__name__)

    def process_file(self, input_path: str, output_path: str) -> Dict:
        """Process a single audio file through the pipeline"""
        self.logger.info("Getting speaker diarization from AssemblyAI...")
        transcript = self.assemblyai.transcribe(input_path)
        
        if transcript.status == aai.TranscriptStatus.error:
            error_msg = f"Transcription failed: {transcript.error}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
            
        if self.whisper:
            # process each utterance with Whisper
            self.logger.info("Enhancing transcription with Whisper...")
            enhanced_utterances = []
            
            for utterance in transcript.utterances:
                try:
                    whisper_result = self.whisper.transcribe_utterance(
                        input_path,
                        start=utterance.start / 1000,  # Convert ms to seconds
                        end=utterance.end / 1000
                    )

                    # create enhanced utterance with Whisper transcription
                    # but keep speaker info from AssemblyAI
                    words = []
                    if self.chat_builder.include_word_timestamps:
                        words = [
                            {
                                'text': w['text'],
                                'start': int(w['start'] * 1000),  # Convert to ms
                                'end': int(w['end'] * 1000),
                                'confidence': utterance.confidence
                            }
                            for w in whisper_result.get('words', [])
                        ]
                    
                    enhanced_utterance = type('EnhancedUtterance', (), {
                        'speaker': utterance.speaker,
                        'start': utterance.start,
                        'end': utterance.end,
                        'confidence': utterance.confidence,
                        'text': whisper_result['text'],
                        'words': words
                    })
                    enhanced_utterances.append(enhanced_utterance)
                except Exception as e:
                    self.logger.warning(f"Error enhancing utterance with Whisper, using AssemblyAI fallback: {e}")
                    enhanced_utterances.append(utterance)
            
            utterances = enhanced_utterances
        else:
            utterances = transcript.utterances
            
        # convert to chat format
        self.logger.info("Generating chat format...")
        chat_data = self.chat_builder.build_chat(utterances)
        
        # add transcription source to metadata
        if self.chat_builder.include_metadata:
            chat_data['metadata']['transcription'] = 'whisper+assemblyai' if self.whisper else 'assemblyai'
        
        # save output
        self._save_output(chat_data, output_path)
        return chat_data

    def _save_output(self, chat_data: Dict, output_path: str):
        """Save chat data to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)