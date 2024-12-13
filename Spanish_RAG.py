import fitz  # PyMuPDF
import numpy as np
import requests
from typing import List, Dict, Tuple, Set
import json
from sklearn.metrics.pairwise import cosine_similarity


class RAGProcessor:
    def __init__(self, pdf_path: str, chunk_size: int = 500, embedding_model: str = "bge-m3"):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.embedding_model = embedding_model
        self.chunks = []
        self.embeddings = []
        self.processed_terms = set()

        # Initialize by processing the document
        self._initialize_document()

    def _initialize_document(self):
        """Process document once at initialization."""
        print("Initializing document processing...")
        text = self._extract_text()
        self.chunks = self._chunk_text(text)
        print(f"Created {len(self.chunks)} chunks")
        self.embeddings = self._create_embeddings()
        print("Created embeddings for all chunks")

    def _extract_text(self) -> str:
        """Extract text from PDF file."""
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into roughly equal chunks based on character count."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding from Ollama API."""
        url = "http://localhost:11434/api/embeddings"
        data = {
            "model": self.embedding_model,
            "prompt": text
        }
        response = requests.post(url, json=data)
        return response.json()["embedding"]

    def _create_embeddings(self) -> List[List[float]]:
        """Create embeddings for text chunks using Ollama."""
        return [self._get_embedding(chunk) for chunk in self.chunks]

    def query_document(self, query: str, k: int = 5, min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        """Query the document using embeddings."""
        query_embedding = self._get_embedding(query)
        query_embedding_np = np.array(query_embedding).reshape(1, -1)
        embeddings_np = np.array(self.embeddings)

        similarities = cosine_similarity(query_embedding_np, embeddings_np)[0]
        most_similar_idxs = np.argsort(similarities)[-k:][::-1]

        filtered_results = [(self.chunks[idx], float(similarities[idx]))
                            for idx in most_similar_idxs
                            if similarities[idx] >= min_similarity]
        return filtered_results


    def _query_llm(self, model: str, prompt: str) -> str:
        """Query Ollama model for generation."""
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=data)
        return response.json()["response"]

    def process_query(self, query: str, llm_model: str = "mistral") -> Dict:
        # Query document
        relevant_chunks = self.query_document(query)

        # Format context from relevant chunks
        context = "\n\n".join([chunk for chunk, score in relevant_chunks])

        # Get initial response
        prompt = f"""
            Context from the document:
            {context}

            Question: {query}

            Based on the context provided above, please answer the question. If the answer cannot be found in the context, please say so.
            """

        response = self._query_llm(llm_model, prompt)

        # Base result
        result = {
            "query": query,
            "relevant_chunks": [(chunk, float(score)) for chunk, score in relevant_chunks],
            "response": response,
            "follow_up_results": []
        }

        return result


def format_results(results: Dict, indent: int = 0) -> str:
    """Format results in a readable way."""
    output = []
    prefix = " " * indent

    output.append(f"{prefix}Query: {results['query']}")
    output.append(f"{prefix}Response: {results['response']}")

    return "\n".join(output)


def main():
    # Initialize the RAG processor once
    processor = RAGProcessor(
        "/Users/jdavies/Downloads/Manual-de-Operacion-Aduanera-MOA-may2022.pdf",
        embedding_model="bge-m3"
    )

    # Process the query
    result = processor.process_query(
        "¿Cuáles son los principales elementos que se revisan durante el reconocimiento aduanero de mercancías en una aduana marítima y qué sistemas se utilizan para este proceso?",
        llm_model="qwen2.5",
        recursive_depth=1
    )

    # Print formatted results
    print(format_results(result))

    question = "What are the main elements that are reviewed during the customs inspection of goods at a maritime customs office and what systems are used for this process?"
    result = processor.process_query(
        query=question,
        llm_model="qwen2.5",
        recursive_depth=1
    )
    print(format_results(result))

if __name__ == "__main__":
    main()