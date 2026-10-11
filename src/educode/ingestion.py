import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple

from educode.io import JSONLSource
from educode.models import AIDeclaration, Submission


def _compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_and_parse(raw_json: str) -> Submission:
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"FORMAT_ERROR: Invalid JSON format: {e}")

    required_fields = {
        "submission_id",
        "student_id",
        "timestamp",
        "code",
        "ai_declaration",
    }
    if not required_fields.issubset(data.keys()):
        raise ValueError(
            f"SCHEMA_VIOLATION: Missing required fields: "
            f"{required_fields - data.keys()}"
        )

    if (
        not isinstance(data.get("ai_declaration"), dict)
        or "used" not in data["ai_declaration"]
    ):
        raise ValueError("SCHEMA_VIOLATION: Invalid ai_declaration structure")

    try:
        ts_str = data["timestamp"].replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            raise ValueError("Timezone is required")
    except Exception as e:
        raise ValueError(f"SCHEMA_VIOLATION: Invalid timestamp: {e}")

    return Submission(
        submission_id=str(data["submission_id"]),
        student_id=str(data["student_id"]),
        timestamp=ts,
        code=str(data["code"]),
        ai_declaration=AIDeclaration(used=bool(data["ai_declaration"]["used"])),
    )


class IngestionService:
    """Processes stream, validates schema, and resolves ID collisions."""

    def __init__(self) -> None:
        self.repository: Dict[str, Tuple[Submission, str]] = {}
        self.accepted: List[Submission] = []
        self.duplicates: List[str] = []
        self.errors: List[Dict[str, Any]] = []

    def process_file(self, filepath: str) -> None:
        with JSONLSource(filepath) as source:
            for line_no, raw_line in source:
                try:
                    submission = validate_and_parse(raw_line)
                    self._handle_submission(line_no, submission)
                except ValueError as e:
                    err_msg = str(e)
                    code, message = (
                        err_msg.split(": ", 1)
                        if ": " in err_msg
                        else ("FORMAT_ERROR", err_msg)
                    )
                    self.errors.append(
                        {"line": line_no, "code": code, "message": message}
                    )

    def _handle_submission(self, line_no: int, submission: Submission) -> None:
        sub_id = submission.submission_id
        code_hash = _compute_hash(submission.code)

        if sub_id in self.repository:
            _, existing_hash = self.repository[sub_id]
            if existing_hash == code_hash:
                self.duplicates.append(sub_id)
            else:
                self.errors.append(
                    {
                        "line": line_no,
                        "code": "ID_CONFLICT",
                        "submission_id": sub_id,
                        "message": "Hash mismatch for existing ID",
                    }
                )
        else:
            self.repository[sub_id] = (submission, code_hash)
            self.accepted.append(submission)
