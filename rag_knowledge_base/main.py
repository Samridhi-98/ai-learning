# main.py
import os
from indexer import index_directory, collection
from rag_chain import ask


def main():
    print("=== Internal Knowledge Base ===\n")

    # Index documents if collection is empty
    if collection.count() == 0:
        print("No documents indexed yet. Indexing ./documents/ folder...")
        os.makedirs("documents", exist_ok=True)
        index_directory("./documents")
    else:
        print(f"Knowledge base loaded — {collection.count()} chunks indexed")
        print("Commands: 'reindex' to re-index docs | 'quit' to exit\n")

    print("\nAsk anything about your documents!\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        elif user_input.lower() == "quit":
            break
        elif user_input.lower() == "reindex":
            index_directory("./documents")
            continue

        print("\nAssistant: ", end="")
        answer = ask(user_input)
        print(answer)
        print()


if __name__ == "__main__":
    main()