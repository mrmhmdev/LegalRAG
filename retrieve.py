import os
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EMBEDDINGS_FILE = "data/embeddings.pkl"
FAISS_INDEX_FILE = "data/faiss_index.bin"
CHUNKS_FILE = "data/chunks.pkl"


class Retriever:
    def __init__(self):
        try:
            logger.info("Initializing Retriever...")
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model")
            
            if not os.path.exists(FAISS_INDEX_FILE):
                logger.error(f"FAISS index not found at {FAISS_INDEX_FILE}")
                raise FileNotFoundError(f"FAISS index not found. Run ingest.py first.")
            
            self.index = faiss.read_index(FAISS_INDEX_FILE)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            with open(CHUNKS_FILE, "rb") as f:
                self.chunks = pickle.load(f)
            logger.info(f"Loaded {len(self.chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to initialize Retriever: {str(e)}")
            raise
    
    def retrieve(self, query, top_k=3):
        try:
            logger.info(f"Retrieving top {top_k} chunks for query: '{query}'")
            query_embedding = self.model.encode([query])[0]
            query_embedding = np.array([query_embedding]).astype(np.float32)
            
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = [self.chunks[idx] for idx in indices[0]]
            logger.info(f"Successfully retrieved {len(results)} chunks")
            for i, (text, dist) in enumerate(zip(results, distances[0])):
                logger.info(f"Rank {i+1}: Chunk Text: `{text[:100]}` | Distance: {dist:.2f}")
            return results
        except Exception as e:
            logger.error(f"Retrieval failed for query '{query}': {str(e)}")
            raise


if __name__ == "__main__":
    try:
        retriever = Retriever()
        query = "What is the main topic?"
        results = retriever.retrieve(query)
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        logger.critical(f"Test retrieval failed: {str(e)}")
