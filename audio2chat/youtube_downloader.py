from pathlib import Path
import yt_dlp


def download_audio(url: str, output_dir: str = "data/raw", audio_format: str = "wav") -> str:
    """
    Download audio from YouTube video
    
    Args:
        url: YouTube video URL
        output_dir: Directory to save downloaded audio
        audio_format: Audio format to download (mp3 or wav)
        
    Returns:
        str: Path to downloaded file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',
        }],
        'outtmpl': str(output_path / '%(title)s.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        filename = str(Path(filename).with_suffix(f'.{audio_format}'))
        return filename