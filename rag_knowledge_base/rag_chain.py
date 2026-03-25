# rag_chain.py
from anthropic import Anthropic
from retriever import retrieve, format_context

client = Anthropic()

# Conversation history — same pattern as your Phase 1 chatbot!
chat_history = []

SYSTEM_PROMPT = """You are a helpful assistant for an internal company knowledge base.
Answer questions using ONLY the provided context chunks.
If the answer isn't in the context, say "I don't have that information in the knowledge base."
Always cite which document your answer came from."""


def rewrite_query(user_query: str) -> str:
    """
    Conversational RAG key step: if there's chat history,
    rewrite the query to be self-contained using prior context.
    """
    # No history yet — first turn, no rewrite needed
    if not chat_history:
        return user_query

    # Build a compact history summary for the rewriter
    history_text = "\n".join([
        f"{msg['role'].upper()}: {msg['content'][:200]}"
        for msg in chat_history[-4:]  # only last 2 turns for efficiency
    ])

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{
            "role": "user",
            "content": f"""Given this conversation history:
{history_text}

Rewrite this follow-up question to be self-contained (resolve pronouns, add context):
"{user_query}"

Return ONLY the rewritten question, nothing else."""
        }]
    )

    rewritten = response.content[0].text.strip()
    print(f"  [Query rewrite: '{user_query}' → '{rewritten}']")
    return rewritten


def ask(user_query: str, top_k: int = 4) -> str:
    """
    Full conversational RAG pipeline:
    rewrite → retrieve → augment → generate
    """
    # Step 1: Rewrite query using chat history (conversational RAG)
    search_query = rewrite_query(user_query)

    # Step 2: Retrieve relevant chunks
    chunks = retrieve(search_query, top_k=top_k)
    if not chunks:
        return "No relevant information found in the knowledge base."

    # Step 3: Augment — build prompt with retrieved context
    context = format_context(chunks)
    augmented_prompt = f"""Answer the question using the context below.

CONTEXT:
{context}

QUESTION: {user_query}"""

    # Step 4: Add to history, generate response
    chat_history.append({"role": "user", "content": augmented_prompt})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=chat_history
    )

    answer = response.content[0].text
    chat_history.append({"role": "assistant", "content": answer})
    return answer