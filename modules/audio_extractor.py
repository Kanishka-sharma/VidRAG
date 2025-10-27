"""
Download YouTube audio and extract as WAV.
"""
from pathlib import Path
from yt_dlp import YoutubeDL

DEFAULT_AUDIO_DIR = Path("data/audio")
DEFAULT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)


def download_audio(youtube_url: str, out_dir: Path = DEFAULT_AUDIO_DIR) -> Path:
    """
    Downloads the YouTube video audio and converts it to WAV.
    Returns path to the .wav file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

   
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(out_dir / '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'noplaylist': True  
    }
    

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get('id')
        filename = out_dir / f"{video_id}.wav"
        return filename


if __name__ == "__main__":
    import sys
    url = sys.argv[1]
    wav_file = download_audio(url)
    print(f"Downloaded audio to: {wav_file}")
