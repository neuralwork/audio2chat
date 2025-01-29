import pytest
from pathlib import Path
from audio2chat.pipeline import AudioChatPipeline


@pytest.fixture
def api_key():
    # this should be set in your environment or CI/CD pipeline
    import os
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    if not api_key:
        pytest.skip("ASSEMBLYAI_API_KEY environment variable not set")
    return api_key

@pytest.fixture
def sample_wav_path():
    path = Path(__file__).parent / "test_data" / "input.wav"
    if not path.exists():
        pytest.skip("Test audio file not found at tests/test_data/input.wav")
    return path

@pytest.fixture
def pipeline(api_key):
    return AudioChatPipeline(
        api_key=api_key,
        language="en",
        num_speakers=2,
        min_segment_confidence=0.5,
        merge_threshold=1.0,
        min_duration=0.5,
        include_metadata=True,
        include_word_timestamps=True,
        use_whisper=True
    )

def test_pipeline_initialization(pipeline):
    assert pipeline.assemblyai is not None
    assert pipeline.whisper is not None
    assert pipeline.chat_builder is not None

def test_process_file(pipeline, sample_wav_path, tmp_path):
    output_path = tmp_path / "output.json"
    chat_data = pipeline.process_file(str(sample_wav_path), str(output_path))
    
    # check output file exists
    assert output_path.exists()
    
    # check basic structure
    assert isinstance(chat_data, dict)
    assert 'messages' in chat_data
    assert 'metadata' in chat_data
    
    # check metadata
    assert 'transcription' in chat_data['metadata']
    assert chat_data['metadata']['transcription'] == 'whisper+assemblyai'
    
    # check messages
    assert isinstance(chat_data['messages'], list)
    if len(chat_data['messages']) > 0:
        message = chat_data['messages'][0]
        assert 'speaker' in message
        assert 'message' in message
        assert 'timestamp' in message
        assert 'start' in message['timestamp']
        assert 'end' in message['timestamp']
        if pipeline.chat_builder.include_word_timestamps:
            assert 'words' in message