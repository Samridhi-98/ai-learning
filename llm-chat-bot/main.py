import json
from anthropic import Anthropic
from BugReport import BugReport

client = Anthropic()

SYSTEM_PROMPT = """You are a helpful assistant for a Java developer 
learning AI engineering. Be concise and use Java analogies where helpful."""
HISTORY_FILE = "chat_history.json"


def extract_bug_report(user_message: str) -> BugReport:
    tools = [
        {
            "name": "extract_bug_report",
            "description": "Extract structured bug report from plain English",
            "input_schema": BugReport.model_json_schema()
        }
    ]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": "extract_bug_report"},
        messages=[{"role": "user", "content": user_message}]
    )

    tool_input = response.content[0].input
    bug_report = BugReport(**tool_input)
    return bug_report


def load_history(filepath: str) -> list[dict[str, str]]:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_history(filepath: str, history: list[dict[str, str]]) -> None:
    with open(filepath, "w") as f:
        json.dump(history, f, indent=4)

conversation_history = load_history(HISTORY_FILE)

def chat(user_message: str) -> str:
    conversation_history.append({"role": "user", "content": user_message})
    response = ""
    print("Assistant:")
    with client.messages.stream(
            max_tokens=1024,
            messages=conversation_history,
            model="claude-haiku-4-5-20251001",
            system=SYSTEM_PROMPT,

    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            response += text
    print("\n")
    assistant_message = response
    conversation_history.append({"role": "assistant", "content": assistant_message})
    save_history(HISTORY_FILE, conversation_history)
    return assistant_message


def main():
    print("=== Streaming Chatbot ===")
    print("Type 'bug: <description>' to extract a structured bug report")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break
        elif user_input.lower().startswith("bug:"):
            description = user_input[4:].strip()
            print("\nExtracting bug report...")
            report = extract_bug_report(description)
            print(f"\n--- Bug Report ---")
            print(f"Title:      {report.title}")
            print(f"Severity:   {report.severity}")
            print(f"Component:  {report.component}")
            print(f"Regression: {report.is_regression}")
            print(f"Steps:")
            for i, step in enumerate(report.reproduction_steps, 1):
                print(f"  {i}. {step}")
            print("------------------\n")
        else:
            chat(user_input)
            print()


if __name__ == "__main__":
    main()