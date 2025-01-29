from datetime import datetime
from typing import List, Dict, Optional, Any


class ChatBuilder:
    def __init__(
        self,
        min_segment_confidence: float = -1.0,
        merge_threshold: float = 1.0,
        min_duration: float = 0.5,
        include_metadata: bool = True,
        include_word_timestamps: bool = False
    ):
        """
        Initialize ChatBuilder with configuration
        
        Args:
            min_segment_confidence: Minimum confidence score to include segment
            merge_threshold: Time threshold (seconds) to merge adjacent utterances
            min_duration: Minimum duration (seconds) for a chat segment
            include_metadata: Whether to include additional metadata in output
            include_word_timestamps: Whether to include word-level timing information
        """
        self.min_segment_confidence = min_segment_confidence
        self.merge_threshold = merge_threshold
        self.min_duration = min_duration
        self.include_metadata = include_metadata
        self.include_word_timestamps = include_word_timestamps

    def build_chat(self, utterances: List[Any]) -> Dict:
        """
        Convert AssemblyAI utterances into chat format
        
        Args:
            utterances: List of AssemblyAI utterance objects
            
        Returns:
            Dict containing structured chat data
        """
        messages = []
        current_speaker = None
        current_message = ""
        current_words = []
        message_start = 0
        message_end = 0
        unique_speakers = []
        
        for utterance in utterances:
            # Skip segments below confidence threshold
            if utterance.confidence < self.min_segment_confidence:
                continue
                
            # Skip segments shorter than minimum duration
            duration = (utterance.end - utterance.start) / 1000  # Convert ms to seconds
            if duration < self.min_duration:
                continue
            
            # Track unique speakers
            if utterance.speaker not in unique_speakers:
                unique_speakers.append(utterance.speaker)

            if (current_speaker is None or 
                utterance.speaker != current_speaker or 
                (utterance.start / 1000 - message_end) > self.merge_threshold):
                
                # Save previous message if exists
                if current_message:
                    messages.append(self._create_message(
                        current_speaker,
                        current_message.strip(),
                        message_start,
                        message_end,
                        current_words if self.include_word_timestamps else None
                    ))
                
                # Start new message
                current_speaker = utterance.speaker
                current_message = utterance.text
                message_start = utterance.start / 1000  # Convert to seconds
                message_end = utterance.end / 1000
                current_words = [dict(word_info) for word_info in utterance.words] if self.include_word_timestamps else []
            else:
                # Continue current message
                current_message += " " + utterance.text
                message_end = utterance.end / 1000
                if self.include_word_timestamps:
                    current_words.extend([dict(word_info) for word_info in utterance.words])

        # Add final message
        if current_message:
            messages.append(self._create_message(
                current_speaker,
                current_message.strip(),
                message_start,
                message_end,
                current_words if self.include_word_timestamps else None
            ))

        result = {'messages': messages}
        
        # Add metadata if enabled
        if self.include_metadata:
            result['metadata'] = {
                'created_at': datetime.now().isoformat(),
                'num_speakers': len(unique_speakers),
                'speaker_ids': unique_speakers,
                'duration': messages[-1]['timestamp']['end'] if messages else 0,
                'word_timestamps': self.include_word_timestamps,
                'config': {
                    'min_segment_confidence': self.min_segment_confidence,
                    'merge_threshold': self.merge_threshold,
                    'min_duration': self.min_duration
                }
            }

        return result

    def _create_message(
        self,
        speaker: str,
        text: str,
        start: float,
        end: float,
        words: Optional[List[Dict]] = None
    ) -> Dict:
        """Helper to create consistent message structure"""
        message = {
            'speaker': speaker,
            'message': text,
            'timestamp': {
                'start': start,
                'end': end
            }
        }
        
        if words:
            # Convert word timestamps from ms to seconds
            converted_words = []
            for word in words:
                converted_words.append({
                    'text': word['text'],
                    'start': word['start'] / 1000,
                    'end': word['end'] / 1000,
                    'confidence': word.get('confidence', 1.0)
                })
            message['words'] = converted_words
            
        return message