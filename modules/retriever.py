from typing import List, Dict
import chromadb

class ChromaWrapper:
    def __init__(self, persist_directory: str = "./data/chroma", collection_name: str = "vidsynth"):
        """
        Wrapper for a persistent Chroma vector store.
        """
        # 1. Use the new PersistentClient
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 2. Use the more robust 'get_or_create_collection' method
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, ids: List[str], texts: List[str], metadatas: List[Dict], embeddings=None):
        """
        Add or update documents (upsert) and optional embeddings to the collection.
        """
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query(self, query_text: str, n_results: int = 5) -> Dict:
        """
        Retrieve top-n similar documents for the query.
        """
        results = self.collection.query(query_texts=[query_text], n_results=n_results)
        # returns dict with ids, documents, metadatas, distances
        return results

