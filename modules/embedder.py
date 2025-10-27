"""
Embedder module: convert text chunks into vector embeddings using Google Gemini.
Fallback: local SentenceTransformers if GOOGLE_API_KEY is not set.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import os
import google.generativeai as genai

# --- Setup Gemini Client ---
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

class Embedder:
    def __init__(self, provider: str = "gemini", model: str = "models/embedding-001"):
        self.provider = provider
        self.model = model  # Store the model name for later use

        if provider == "gemini":
            if GEMINI_KEY is None:
                raise ValueError("GOOGLE_API_KEY not set. Please set it in .env")
            self.client = None
        else:
            # Local fallback
            self.client = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Return vector embeddings for a list of texts.
        """
        if self.provider == "gemini":
            # Gemini embeddings API call
            result = genai.embed_content(
                model=self.model,
                content=texts,
                task_type="retrieval_document" # Use 'retrieval_document' for texts to be stored in a vector DB
            )
            return result['embedding']
        else:
            # Local fallback 
            return [v.tolist() for v in self.client.encode(texts)]