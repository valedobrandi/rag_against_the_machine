import bm25s
import Stemmer
import pickle
from typing import List, Dict
from .models import MinimalSource, MinimalSearchResults
from pathlib import Path

class BM25Retriever:
    def __init__(self):
        self.retriever = None
        self.corpus = [] # This will store the actual text of chunks
        self.metadata = [] # This will store the MinimalSource for each chunk

    def build_index(self, chunk_data: List[Dict]):
        """
        Processes chunks and builds the BM25 index.
        chunk_data is a list of dicts: {'content': str, 'metadata': MinimalSource}
        """
        self.corpus = [c['content'] for c in chunk_data]
        self.metadata = [c['metadata'] for c in chunk_data]
        
        # Tokenize the corpus
        # Using a stemmer helps "running" match with "run"
        stemmer = Stemmer.Stemmer("english")
        corpus_tokens = bm25s.tokenize(self.corpus, stemmer=stemmer)
        
        # Create and index the BM25 model
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def save(self, path: str):
        """Saves the index to disk for fast loading."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.retriever.save(str(save_path), corpus=self.corpus)
        metadata_path = save_path / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self, directory: str):
        """
        Loads the index and metadata from disk
        """

        self.retriever = bm25s.BM25.load(directory, load_corpus=True)

        metadata_path = Path(directory) / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Performs search and returns the top-k sources."""
        if not self.retriever:
            raise ValueError("BM25 index not built or loaded. Call build_index() or load() first.")
        stemmer = Stemmer.Stemmer("english")
        query_tokens = bm25s.tokenize(query, stemmer=stemmer)
        
        # Get the top k results
        results, _ = self.retriever.retrieve(query_tokens, k=k)
        # Map the indices of the results back to our metadata
        return [self.metadata[r["id"]] for r in results[0]]