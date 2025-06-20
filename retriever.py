import faiss, pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path, chunks_path):
        self.index = faiss.read_index(index_path)
        self.chunks = pickle.load(open(chunks_path, "rb"))
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def get_relevant_chunks(self, query, k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]
