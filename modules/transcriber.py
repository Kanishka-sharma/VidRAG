"""
Transcription using faster-whisper.
"""
from pathlib import Path
from faster_whisper import WhisperModel
from typing import Dict

DEFAULT_TRANSCRIPT_DIR = Path("data/transcripts")
DEFAULT_TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)


class Transcriber:
    def __init__(self, model_size: str = "small", device: str = "cpu"):
        """
        Initialize Whisper transcription model.
        """
        self.model = WhisperModel(model_size, device=device)

    def transcribe(self, audio_path: str, progress: bool = True) -> Dict:
        """
        Transcribe audio file to segments.
        Returns dictionary
        """
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        segment_list = []
        for seg in segments:
            segment_list.append({
                'start': seg.start,
                'end': seg.end,
                'text': seg.text.strip()
            })
        return {
            'language': info.language,
            'segments': segment_list
        }


def save_transcript(transcript: Dict, output_path: Path):
    """
    Save transcript dictionary to JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import sys
    audio_file = sys.argv[1]
    transcriber = Transcriber(model_size="small", device="cpu")
    result = transcriber.transcribe(audio_file)
    out_file = DEFAULT_TRANSCRIPT_DIR / (Path(audio_file).stem + ".json")
    save_transcript(result, out_file)
    print(f"Saved transcript to: {out_file}")
