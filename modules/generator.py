"""
RAG generator: retrieve chunks and call Gemini LLM for grounded answers.
"""
from typing import List, Dict
import os
import google.generativeai as genai

# --- Setup Gemini Client ---
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")

if GEMINI_KEY is None:
    raise ValueError("GOOGLE_API_KEY not set. Please add it to your .env file")

genai.configure(api_key=GEMINI_KEY)

class RAGGenerator:
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.0):
        """
        Initialize Gemini LLM for RAG generation.
        """
        self.model_name = model_name
        self.temperature = temperature
        # Initialize the actual Gemini client and generation config
        self.client = genai.GenerativeModel(self.model_name)
        self.generation_config = genai.types.GenerationConfig(temperature=self.temperature)

    def make_prompt(self, question: str, retrieved_chunks: List[Dict]) -> str:
        """
        Build prompt combining retrieved chunks and user question.
        """
        context = "\n\n---\n\n".join([f"[{i}] {c['text']}" for i, c in enumerate(retrieved_chunks)])
        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the provided context. "
            "Cite chunk indices in square brackets when you use them. "
            "If the answer is not contained in the context, respond honestly that you don't know.\n\n"
            f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer:"
        )
        return prompt

    def answer(self, question: str, retrieved_chunks: List[Dict]) -> Dict:
        """
        Generate answer from Gemini LLM based on retrieved chunks.
        """
        prompt = self.make_prompt(question, retrieved_chunks)
        response = self.client.generate_content(
            prompt,
            generation_config=self.generation_config
        )

        answer_text = response.text
        return {
            "answer": answer_text,
            "llm_response_raw": response
        }