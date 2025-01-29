import os
import time
import requests
from typing import Dict, Optional, Union
from pathlib import Path


class AssemblyAIClient:
    """Client for interacting with the AssemblyAI API"""
    
    def __init__(self, api_key: str):
        """
        Initialize AssemblyAI client
        
        Args:
            api_key: AssemblyAI API key
        """
        self.api_key = api_key
        self.base_url = "https://api.assemblyai.com/v2"
        self.headers = {
            "authorization": api_key,
            "content-type": "application/json"
        }

    def upload_file(self, file_path: Union[str, Path]) -> str:
        """
        Upload an audio file to AssemblyAI
        
        Args:
            file_path: Path to audio file
            
        Returns:
            str: URL of uploaded file
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        upload_url = f"{self.base_url}/upload"
        
        def read_file(file_path):
            with open(file_path, "rb") as f:
                while True:
                    data = f.read(5242880)  # Read in 5MB chunks
                    if not data:
                        break
                    yield data

        upload_response = requests.post(
            upload_url,
            headers={"authorization": self.api_key},
            data=read_file(file_path)
        )
        
        if upload_response.status_code != 200:
            raise Exception(f"Upload failed: {upload_response.text}")
            
        return upload_response.json()["upload_url"]

    def create_transcript(
        self,
        audio_url: str,
        speaker_labels: bool = True,
        speakers_expected: Optional[int] = None,
        word_boost: Optional[list] = None
    ) -> str:
        """
        Submit audio for transcription
        
        Args:
            audio_url: URL of uploaded audio file
            speaker_labels: Enable speaker diarization
            speakers_expected: Expected number of speakers
            word_boost: List of words to boost recognition for
            
        Returns:
            str: ID of transcription job
        """
        data = {
            "audio_url": audio_url,
            "speaker_labels": speaker_labels,
            "speakers_expected": speakers_expected,
            "word_boost": word_boost
        }
        
        response = requests.post(
            f"{self.base_url}/transcript",
            json=data,
            headers=self.headers
        )
        
        if response.status_code != 200:
            raise Exception(f"Transcription request failed: {response.text}")
            
        return response.json()["id"]

    def get_transcript(self, transcript_id: str) -> Dict:
        """
        Get transcription results
        
        Args:
            transcript_id: ID of transcription job
            
        Returns:
            Dict: Transcription results including utterances
        """
        polling_endpoint = f"{self.base_url}/transcript/{transcript_id}"
        
        while True:
            response = requests.get(polling_endpoint, headers=self.headers)
            response = response.json()

            if response["status"] == "completed":
                return response
            elif response["status"] == "error":
                raise Exception(f"Transcription failed: {response['error']}")
                
            time.sleep(3)

    def transcribe(
        self,
        file_path: Union[str, Path],
        speaker_labels: bool = True,
        speakers_expected: Optional[int] = None,
        word_boost: Optional[list] = None
    ) -> Dict:
        """
        Complete transcription pipeline: upload file and get results
        
        Args:
            file_path: Path to audio file
            speaker_labels: Enable speaker diarization
            speakers_expected: Expected number of speakers
            word_boost: List of words to boost recognition for
            
        Returns:
            Dict: Transcription results including utterances
        """
        audio_url = self.upload_file(file_path)
        transcript_id = self.create_transcript(
            audio_url,
            speaker_labels,
            speakers_expected,
            word_boost
        )
        return self.get_transcript(transcript_id)