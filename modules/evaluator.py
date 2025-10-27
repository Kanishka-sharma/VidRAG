"""
Evaluation utilities: compute simple RAGAS-like metrics and custom metrics.
"""
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.translate.bleu_score import sentence_bleu
import re

nltk.download('punkt', quiet=True)


def semantic_similarity(a: str, b: str) -> float:
    """
    Compute lightweight TF-IDF cosine similarity between two texts.
    """
    vect = TfidfVectorizer().fit([a, b])
    m = vect.transform([a, b])
    sim = cosine_similarity(m[0], m[1])[0][0]
    return float(sim)


def faithfulness_score(answer: str, retrieved_contexts: List[str]) -> float:
    """
    Approximate faithfulness: max similarity between answer and any context chunk.
    """
    sims = [semantic_similarity(answer, c) for c in retrieved_contexts]
    return float(max(sims))


def relevance_score(answer: str, question: str) -> float:
    """
    BLEU score as a proxy for relevance of answer to question.
    """
    try:
        ref = nltk.word_tokenize(question)
        cand = nltk.word_tokenize(answer)
        return float(sentence_bleu([ref], cand))
    except Exception:
        return 0.0


def timestamp_accuracy(answer: str, chunks_meta: List[Dict]) -> float:
    idxs = re.findall(r"\[(\d+)\]", answer)
    if not idxs:
        return 0.0
    idxs = [int(i) for i in idxs]
    valid = 0
    for i in idxs:
        if 0 <= i < len(chunks_meta):
            valid += 1
    return valid / len(idxs)


def aggregate_metrics(answer: str, question: str, retrieved_chunks: List[Dict]) -> Dict:

    contexts = [c['text'] for c in retrieved_chunks]
    metas = [c.get('meta', {}) for c in retrieved_chunks]
    return {
        "faithfulness": faithfulness_score(answer, contexts),
        "relevance": relevance_score(answer, question),
        "timestamp_accuracy": timestamp_accuracy(answer, metas),
        "semantic_coherence": float(np.mean([semantic_similarity(answer, c) for c in contexts]))
    }
