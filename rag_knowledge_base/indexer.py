# indexer.py
import os
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Models ──────────────────────────────────────────────────────────────────
# Local embedding model — free, no API key needed
# all-MiniLM-L6-v2: 384 dimensions, fast, good quality for knowledge bases
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ── ChromaDB client (persistent — survives restarts) ─────────────────────────
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}   # use cosine similarity, not euclidean
)

# ── Text splitter ────────────────────────────────────────────────────────────
# RecursiveCharacterTextSplitter tries to split on paragraphs first,
# then sentences, then words — respects natural text boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # ~500 tokens per chunk
    chunk_overlap=100,    # 100 token overlap between chunks
    separators=["\n\n", "\n", ". ", " "]  # priority order of split points
)


def load_pdf(filepath: str) -> str:
    """Extract raw text from a PDF file."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def load_txt(filepath: str) -> str:
    """Load plain text file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def index_document(filepath: str) -> int:
    """
    Full indexing pipeline for one document:
    load → split → embed → store in ChromaDB
    Returns number of chunks created.
    """
    path = Path(filepath)
    print(f"Indexing: {path.name}")

    # Step 1: Load raw text based on file type
    if path.suffix == ".pdf":
        text = load_pdf(filepath)
    elif path.suffix == ".txt":
        text = load_txt(filepath)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    # Step 2: Split into chunks
    chunks = splitter.split_text(text)
    print(f"  Created {len(chunks)} chunks")

    # Step 3: Embed all chunks in one batch (faster than one by one)
    embeddings = EMBEDDING_MODEL.encode(chunks).tolist()

    # Step 4: Store in ChromaDB with metadata
    # Each chunk gets: unique id, the text, its vector, and source metadata
    collection.add(
        ids=[f"{path.stem}_chunk_{i}" for i in range(len(chunks))],
        documents=chunks,                          # raw text (for LLM context)
        embeddings=embeddings,                     # vectors (for similarity search)
        metadatas=[{
            "source": path.name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]
    )

    print(f"  Stored in ChromaDB — collection size: {collection.count()}")
    return len(chunks)


def index_directory(directory: str) -> None:
    """Index all supported documents in a directory."""
    supported = {".pdf", ".txt"}
    files = [f for f in Path(directory).iterdir() if f.suffix in supported]

    if not files:
        print(f"No supported files found in {directory}")
        return

    total_chunks = 0
    for filepath in files:
        total_chunks += index_document(str(filepath))

    print(f"\nIndexing complete. Total chunks: {total_chunks}")