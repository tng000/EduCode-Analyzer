from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AIDeclaration:
    used: bool


@dataclass(frozen=True)
class Submission:
    submission_id: str
    student_id: str
    timestamp: datetime
    code: str
    ai_declaration: AIDeclaration


@dataclass(frozen=True)
class Assignment:
    assignment_id: str
    permitted_templates: List[str] = field(default_factory=list)


@dataclass
class AnalysisReport:
    analyzer_id: str
    analyzer_version: str
    score: Optional[int] = None
    similarity: Optional[float] = None
    rubric_feedback: str = ""
    evidence: List[Dict[str, Any]] = field(default_factory=list)
