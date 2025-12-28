import os
import pickle
import logging
from pathlib import Path
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = "data"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.pkl")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.pkl")


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def extract_text_from_pdf(pdf_path):
    try:
        logger.info(f"Extracting text from {pdf_path}")
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        logger.info(f"Successfully extracted {len(text)} characters from {pdf_path}")
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {str(e)}")
        raise


def ingest_documents(pdf_folder="data"):
    try:
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning("No PDF files found in data folder.")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded SentenceTransformer model")
        
        all_chunks = []
        all_embeddings = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Processing {pdf_file.name}...")
                text = extract_text_from_pdf(pdf_file)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                logger.info(f"Created {len(chunks)} chunks from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                continue
        
        if not all_chunks:
            logger.error("No chunks were created from any PDF files")
            return
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        logger.info("Generating embeddings...")
        embeddings = model.encode(all_chunks, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        os.makedirs(DATA_DIR, exist_ok=True)
        
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(all_chunks, f)
        logger.info(f"Saved chunks to {CHUNKS_FILE}")
        
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"Saved embeddings to {EMBEDDINGS_FILE}")
        
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, FAISS_INDEX_FILE)
        logger.info(f"Saved FAISS index to {FAISS_INDEX_FILE}")
        
        logger.info(f"Ingestion complete. Index contains {len(all_chunks)} chunks.")
    
    except Exception as e:
        logger.critical(f"Ingestion pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    ingest_documents()
