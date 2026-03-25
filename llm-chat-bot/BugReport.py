from pydantic import BaseModel
from typing import Literal

class BugReport(BaseModel):
    title: str
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    component: str
    reproduction_steps: list[str]
    is_regression: bool
