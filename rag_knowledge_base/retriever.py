# retriever.py
from indexer import EMBEDDING_MODEL, collection


def retrieve(query: str, top_k: int = 4, source_filter: str = None) -> list[dict]:
    """
    Convert query to vector, find top-k similar chunks in ChromaDB.
    Optionally filter by source document.
    """
    # Step 1: Embed the query using the SAME model used during indexing
    # Critical: must use identical model or similarity scores are meaningless
    query_vector = EMBEDDING_MODEL.encode(query).tolist()

    # Step 2: Build optional metadata filter
    where_clause = {"source": source_filter} if source_filter else None

    # Step 3: Similarity search
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where_clause,            # metadata filtering
        include=["documents", "metadatas", "distances"]
    )

    # Step 4: Format results
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "similarity": round(1 - results["distances"][0][i], 3),  # distance→similarity
            "chunk_index": results["metadatas"][0][i]["chunk_index"]
        })

    return chunks


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a clean context block for the LLM prompt."""
    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"[Source: {chunk['source']} | Similarity: {chunk['similarity']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)