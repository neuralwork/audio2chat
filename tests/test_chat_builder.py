import pytest
from audio2chat.chat_builder import ChatBuilder
from types import SimpleNamespace


@pytest.fixture
def chat_builder():
    return ChatBuilder(
        min_segment_confidence=0.5,
        merge_threshold=1.0,
        min_duration=0.5,
        include_metadata=True,
        include_word_timestamps=True
    )

def test_build_chat(chat_builder):
    # create mock utterances that mimic AssemblyAI/Whisper output
    utterances = [
        SimpleNamespace(
            speaker='A',
            text='Hello there',
            start=0,
            end=1000, # in ms
            confidence=0.9,
            words=[
                {'text': 'Hello', 'start': 0, 'end': 500, 'confidence': 0.9},
                {'text': 'there', 'start': 500, 'end': 1000, 'confidence': 0.9}
            ]
        ),
        SimpleNamespace(
            speaker='B',
            text='Hi, how are you?',
            start=1500,
            end=2500,
            confidence=0.95,
            words=[
                {'text': 'Hi', 'start': 1500, 'end': 1700, 'confidence': 0.95},
                {'text': 'how', 'start': 1700, 'end': 2000, 'confidence': 0.95},
                {'text': 'are', 'start': 2000, 'end': 2200, 'confidence': 0.95},
                {'text': 'you', 'start': 2200, 'end': 2500, 'confidence': 0.95}
            ]
        )
    ]
    
    chat_data = chat_builder.build_chat(utterances)
    
    # check structure
    assert isinstance(chat_data, dict)
    assert 'messages' in chat_data
    assert 'metadata' in chat_data
    
    # check metadata
    assert chat_data['metadata']['num_speakers'] == 2
    assert chat_data['metadata']['speaker_ids'] == ['A', 'B']
    
    # check messages
    messages = chat_data['messages']
    assert len(messages) == 2
    
    # check first message
    assert messages[0]['speaker'] == 'A'
    assert messages[0]['message'] == 'Hello there'
    assert messages[0]['timestamp']['start'] == 0
    assert messages[0]['timestamp']['end'] == 1.0
    
    # check word timestamps if enabled
    if chat_builder.include_word_timestamps:
        assert 'words' in messages[0]
        assert len(messages[0]['words']) == 2
        assert messages[0]['words'][0]['text'] == 'Hello'