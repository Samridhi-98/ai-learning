class AssistantMessage:
    title: str
    content: str
    code: str

    def __init__(self, title: str, content: str, code: str):
        self.title = title
        self.content = content
        self.code = code

    def __str__(self):
        return f"{self.title}\n{self.content}\n{self.code}"