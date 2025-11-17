from pydantic import BaseModel
from typing import List, Optional

class Student(BaseModel):
    id: Optional[str] = None
    name: str
    class_id: Optional[str] = None
    completed_chapters: List[str] = []
