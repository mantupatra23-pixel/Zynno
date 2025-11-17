from pydantic import BaseModel, Field
from typing import Optional

class User(BaseModel):
    id: Optional[str] = None
    name: str
    email: str
    role: str  # "student" or "teacher"
