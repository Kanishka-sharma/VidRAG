# 🎬 VidRAG - Video Understanding with Retrieval-Augmented Generation

**VidRAG** is an intelligent pipeline that lets you **ask questions about YouTube videos** using **Retrieval-Augmented Generation (RAG)** powered by **Google Gemini**.  
It automatically extracts audio, transcribes speech, creates semantic chunks, embeds them into a vector database, and generates **context-grounded answers** - complete with evaluation metrics for trustworthiness and relevance.

---

## Key Features

- **Audio Extraction** - Download and convert YouTube audio automatically  
- **Speech-to-Text** - Transcribe videos with `faster-whisper`  
- **Chunking & Preprocessing** - Split transcripts into semantically meaningful pieces  
- **RAG Pipeline** - Retrieve relevant chunks and generate answers using Gemini  
- **Evaluation Metrics** - Assess answer *faithfulness*, *relevance*, and *semantic coherence*  
- **Streamlit Dashboard** - Intuitive UI for exploring and querying video content  
- **CLI Automation** - Run the full pipeline from your terminal

---

## Tech Stack

| Component | Tool / Library |
|------------|----------------|
| **LLM** | Google Gemini 2.5 Flash |
| **Speech Recognition** | faster-whisper |
| **Vector Database** | ChromaDB |
| **Embeddings** | Gemini or SentenceTransformers |
| **Evaluation** | TF-IDF, BLEU, cosine similarity |
| **Interface** | Streamlit |
| **Language** | Python 3.9+ |

---
## How It Works

- Extract Audio from YouTube using yt-dlp
- Transcribe Speech into text with timestamps via faster-whisper
- Chunk & Embed transcript sections for semantic retrieval
- Store embeddings in ChromaDB for vector similarity search
- Retrieve & Generate grounded answers via Gemini
- Evaluate responses using faithfulness, relevance, and coherence scores

## Evaluation Metrics

| **Metric** | **Description** |
|-------------|-----------------|
| **Faithfulness** | How well the answer aligns with the retrieved chunks |
| **Relevance** | How relevant the answer is to the question |
| **Semantic Coherence** | Measures the similarity between the answer and the retrieved context |
| **Timestamp Accuracy** | Checks whether the cited context indices in the answer are valid |
