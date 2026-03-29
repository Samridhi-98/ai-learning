from mcp.server.fastmcp import FastMCP

from indexer import load_pdf
from rag_chain import SYSTEM_PROMPT
from retriever import retrieve

mcp = FastMCP("rag-knowledge-base")

@mcp.tool()
def search_knowledge_base(query: str) -> str:
    """
    Search the knowledge base for information.
    """
    search_data_set = retrieve(query)
    return search_data_set

@mcp.resource("knowledge://documents/handbook.pdf")
def get_handbook() -> str:
    """
    Get the handbook document.
    """
    return load_pdf("rag_knowledge_base/documents/handbook.pdf")

@mcp.prompt()
def rag_system_prompt() -> str:
    """
    Get the system prompt.:
    """
    return SYSTEM_PROMPT


if __name__ == "__main__":
    mcp.run()