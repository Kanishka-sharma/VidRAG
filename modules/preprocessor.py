"""
Preprocessing and chunking of transcripts.
"""
from pathlib import Path
from typing import List, Dict
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_transcript(path: Path) -> Dict:
    """
    Load transcript JSON from file.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def segments_to_documents(transcript: Dict) -> List[Dict]:
    """
    Convert transcript segments into document dicts.
    Each document has 'text', 'start', 'end'.
    """
    docs = []
    for seg in transcript['segments']:
        docs.append({
            'text': seg['text'],
            'start': seg['start'],
            'end': seg['end']
        })
    return docs

def chunk_documents(docs: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    Chunk text into semantic pieces using RecursiveCharacterTextSplitter.
    Returns a list of chunks with id, text, and meta.
    """
    texts = [d['text'] for d in docs]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "]
    )
    raw_chunks = splitter.split_text("\n\n".join(texts))

    # Map back approximate metadata
    chunks = []
    for i, chunk in enumerate(raw_chunks):
        chunks.append({
            'id': f'chunk_{i}',
            'text': chunk,
            'meta': {}  
        })
    return chunks


if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1])
    transcript = load_transcript(path)
    docs = segments_to_documents(transcript)
    chunks = chunk_documents(docs)
    out_file = Path("data/transcripts") / (path.stem + ".chunks.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
    print(f"Saved {len(chunks)} chunks to {out_file}")
