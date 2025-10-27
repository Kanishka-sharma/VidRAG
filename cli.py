import argparse
from pathlib import Path
import os
from modules.audio_extractor import download_audio
from modules.transcriber import Transcriber, save_transcript
from modules.preprocessor import segments_to_documents, chunk_documents
from modules.embedder import Embedder
from modules.retriever import ChromaWrapper
from modules.generator import RAGGenerator
from modules.evaluator import aggregate_metrics
from modules.utils import save_json

def run_pipeline(youtube_url: str, workdir: Path = Path(".")):
    # Download audio
    wav = download_audio(youtube_url, out_dir=workdir / "audio")

    # Transcribe
    transcriber = Transcriber(model_size="small", device="cpu")
    transcript = transcriber.transcribe(str(wav))
    transcript_path = workdir / "transcripts" / (wav.stem + ".json")
    save_transcript(transcript, transcript_path)

    # Chunk transcripts
    docs = segments_to_documents(transcript)
    chunks = chunk_documents(docs)

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c.get("meta", {}) for c in chunks]

    # Generate embeddings
    provider = "gemini" if os.getenv("GOOGLE_API_KEY") else "sbert"
    emb = Embedder(provider=provider)
    embeddings = emb.embed_texts(texts)

    # Index into Chroma
    db = ChromaWrapper(persist_directory=str(workdir / "chroma"))
    db.add_documents(ids=ids, texts=texts, metadatas=metas, embeddings=embeddings)

    # Simple retrieval and RAG generation demo
    retr = db.query("What is discussed about deep learning?", n_results=5)
    retrieved_chunks = []
    for id_val, doc_val in zip(retr["ids"][0], retr["documents"][0]):
        retrieved_chunks.append({"id": id_val, "text": doc_val, "meta": {}})

    # Generate answer
    generator = RAGGenerator(model_name="gemini-2.5-flash")
    answer = generator.answer("What is the main message of the video?", retrieved_chunks)

    # Evaluate
    metrics = aggregate_metrics(
        answer["answer"],
        "What is the main message of the video?",
        retrieved_chunks
    )

    # Save evaluation
    eval_dir = workdir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / (wav.stem + ".eval.json")
    save_json({"answer": answer, "metrics": metrics}, eval_path)
    print(f"Pipeline completed. Evaluation saved at {eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VidRAG pipeline")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--workdir", default="data", help="Working directory for storing data")
    args = parser.parse_args()

    run_pipeline(args.url, Path(args.workdir))
