from pydantic import BaseModel
from typing import Optional, List

class Classroom(BaseModel):
    id: Optional[str] = None
    class_name: str
    subject: str
    teacher_id: str
    students: List[str] = []
