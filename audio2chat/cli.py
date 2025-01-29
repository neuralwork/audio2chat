import re
import logging
import argparse
from pathlib import Path

from .pipeline import AudioChatPipeline
from .youtube_downloader import download_audio


def is_youtube_url(url: str) -> bool:
    youtube_regex = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    return bool(re.match(youtube_regex, url))

def main():
    parser = argparse.ArgumentParser(description='Convert multi-speaker audio to chat format')
    
    # required arguments
    parser.add_argument('input', type=str, help='Input audio file path or YouTube URL')
    parser.add_argument('--api-key', type=str, required=True,
                       help='Assembly AI API key, required for transcription and speaker diarization')
    
    # output settings
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for audio and chat data (default: output)')
    parser.add_argument('--download-format', type=str, choices=['mp3', 'wav'], default='wav',
                       help='Audio format for YouTube downloads (default: wav)')
    
    # transcription settings
    parser.add_argument('--language', type=str, default='en',
                       help='Language code for transcription (default: en)')
    parser.add_argument('--num-speakers', type=int, default=None,
                       help='Expected number of speakers (default: auto-detect)')
    
    # chat generation settings
    parser.add_argument('--min-segment-confidence', type=float, default=0.5,
                       help='Minimum confidence score to include segment (default: 0.5)')
    parser.add_argument('--merge-threshold', type=float, default=1.0,
                       help='Time threshold in seconds to merge adjacent utterances (default: 1.0)')
    parser.add_argument('--min-duration', type=float, default=0.5,
                       help='Minimum duration in seconds for a chat segment (default: 0.5)')
    parser.add_argument('--include-metadata', action='store_true', default=True,
                       help='Include additional metadata in output (default: True)')
    parser.add_argument('--include-word-timestamps', action='store_true', default=False,
                       help='Include word-level timing information (default: True)')
    
    # vocabulary settings
    parser.add_argument('--word-boost', type=str, nargs='*', help='List of words to boost recognition for')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    # whisper settings
    parser.add_argument('--use-whisper', action='store_true', default=False,
                       help='Use Whisper for enhanced transcription (default: True)')
    
    args = parser.parse_args()

    # set up logging
    log_level = logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    try:        
        input_path = args.input
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        if is_youtube_url(input_path):
            logger.info(f"Downloading audio from YouTube: {input_path}")
            input_path = download_audio(
                url=input_path,
                output_dir=output_dir / 'audio',
                audio_format=args.download_format
            )

        # set up output JSON path
        output_json = output_dir / f"{Path(input_path).stem}_chat.json"

        pipeline = AudioChatPipeline(
            api_key=args.api_key,
            language=args.language,
            num_speakers=args.num_speakers,
            word_boost=args.word_boost,
            min_segment_confidence=args.min_segment_confidence,
            merge_threshold=args.merge_threshold,
            min_duration=args.min_duration,
            include_metadata=args.include_metadata,
            include_word_timestamps=args.include_word_timestamps,
            use_whisper=args.use_whisper
        )
        
        chat_data = pipeline.process_file(input_path, str(output_json))
        logger.info(f"Successfully processed to {output_json}")

        if args.verbose:
            from pprint import pprint
            pprint(chat_data)
    
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise