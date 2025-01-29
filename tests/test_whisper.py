import pytest
from pathlib import Path
from audio2chat.transcriber_whisper import WhisperTranscriber


@pytest.fixture
def sample_wav_path():
    path = Path(__file__).parent / "test_data" / "input.wav"
    if not path.exists():
        pytest.skip("Test audio file not found at tests/test_data/input.wav")
    return path

@pytest.fixture
def transcriber():
    return WhisperTranscriber(
        model_id="openai/whisper-large-v3-turbo",
        language="en"
    )

def test_transcribe_utterance(transcriber, sample_wav_path):
    # test transcribing a short segment
    result = transcriber.transcribe_utterance(
        str(sample_wav_path),
        start=0.0,
        end=2.0
    )
    
    # check structure
    assert isinstance(result, dict)
    assert 'text' in result
    assert 'words' in result
    
    # check text
    assert isinstance(result['text'], str)
    assert len(result['text']) > 0
    
    # check words
    assert isinstance(result['words'], list)
    if len(result['words']) > 0:
        word = result['words'][0]
        assert 'text' in word
        assert 'start' in word
        assert 'end' in word
        assert word['start'] >= 0
        assert word['end'] > word['start']

def test_transcribe_file(transcriber, sample_wav_path):
    result = transcriber.transcribe_file(str(sample_wav_path))
    
    assert isinstance(result, dict)
    assert 'text' in result
    assert isinstance(result['text'], str)
    assert len(result['text']) > 0 