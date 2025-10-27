import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from modules.audio_extractor import download_audio
from modules.transcriber import Transcriber, save_transcript
from modules.preprocessor import segments_to_documents, chunk_documents
from modules.embedder import Embedder
from modules.retriever import ChromaWrapper
from modules.generator import RAGGenerator
from modules.evaluator import aggregate_metrics
from modules.utils import save_json


st.set_page_config(page_title="VidRAG", layout="wide")
st.title("🎬 VidRAG — YouTube Video Q&A & Evaluation")

# This creates a 'memory' for the app.
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'wav_path' not in st.session_state:
    st.session_state.wav_path = None
if 'db' not in st.session_state:
    st.session_state.db = None

# --- Main UI ---
url = st.text_input("Enter YouTube URL")

if st.button("Process Video") and url:
    with st.spinner("Downloading audio..."):
        wav_path = download_audio(url)
        st.session_state.wav_path = wav_path  # Save path to memory
    st.success(f"Audio downloaded: {wav_path.name}")

    with st.spinner("Transcribing..."):
        # Using a smaller, faster model for better user experience
        transcriber = Transcriber(model_size="tiny", device="cpu")
        transcript = transcriber.transcribe(str(wav_path))
        transcript_path = Path("data/transcripts") / (wav_path.stem + ".json")
        save_transcript(transcript, transcript_path)
    st.write("Transcript saved.")

    # Chunking
    docs = segments_to_documents(transcript)
    chunks = chunk_documents(docs)
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c.get("meta", {}) for c in chunks]

    # Embeddings
    provider = "gemini" if os.getenv("GOOGLE_API_KEY") else "sbert"
    embedder = Embedder(provider=provider)
    embeddings = embedder.embed_texts(texts)

    # Vector store
    db = ChromaWrapper(persist_directory="./data/chroma")
    db.add_documents(ids=ids, texts=texts, metadatas=metas, embeddings=embeddings)
    st.session_state.db = db  # Save database connection to memory
    st.success("Video indexed into vector store.")
    
    # Set the memory flag
    st.session_state.processing_complete = True
    st.rerun() # Rerun the script to immediately show the Q&A section

# Decoupled Q&A Section
if st.session_state.processing_complete:
    st.subheader("Ask a Question About the Video")
    query = st.text_input("Enter your question here")

    if st.button("Ask") and query:
        db = st.session_state.db # Load database from memory
        
        with st.spinner("Retrieving relevant chunks..."):
            results = db.query(query, n_results=5)
            retrieved_chunks = []
            # Check if results are valid before trying to access them
            if results and results.get("ids") and results["ids"][0]:
                for id_val, doc_val in zip(results["ids"][0], results["documents"][0]):
                    retrieved_chunks.append({"id": id_val, "text": doc_val})
            else:
                st.warning("Could not retrieve any relevant chunks.")

        st.write("### Retrieved Context:")
        for i, c in enumerate(retrieved_chunks):
            st.markdown(f"**[{i}]** {c['text'][:400]}...")

        # Generate answer
        if retrieved_chunks:
            generator = RAGGenerator()
            with st.spinner("Generating answer..."):
                answer_out = generator.answer(query, retrieved_chunks)
            st.write("### Answer:")
            st.write(answer_out["answer"])

            # Evaluate
            metrics = aggregate_metrics(answer_out["answer"], query, retrieved_chunks)
            st.write("### Metrics:")
            st.json(metrics)

            # Save evaluation
            wav_path = st.session_state.wav_path # Load wav_path from memory
            save_json(
                {"query": query, "answer": answer_out["answer"], "metrics": metrics},
                Path("data/evaluations") / (wav_path.stem + ".eval.json")
            )
            st.success("Evaluation saved.")