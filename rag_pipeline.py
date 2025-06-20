from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self, index_path, chunks_path):
        self.retriever = Retriever(index_path, chunks_path)
        self.generator = Generator()

    def ask(self, query):
        chunks = self.retriever.get_relevant_chunks(query)
        context = "\n\n".join(chunks)
        answer = self.generator.generate(context, query)
        return answer, chunks


