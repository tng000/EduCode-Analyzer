import hashlib
import json
from datetime import datetime
from typing import Any, Dict, Generator, List, Tuple

from educode.io import JSONLSource
from educode.models import AIDeclaration, Assignment, Submission


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


def read_lines(filepath: str) -> Generator[Tuple[int, str], None, None]:
    """Generator: read non-empty lines from a JSONL file."""
    with JSONLSource(filepath) as source:
        for line_no, raw_line in source:
            yield line_no, raw_line


def parse_submission(
    line_no: int, raw_line: str
) -> Tuple[int, Submission, None] | Tuple[int, None, Dict[str, Any]]:
    """Parse a single JSON line, returning (line_no, submission, error).

    On success: (line_no, Submission, None)
    On failure: (line_no, None, error_dict)
    """
    try:
        submission = validate_and_parse(raw_line)
        return line_no, submission, None
    except ValueError as e:
        err_msg = str(e)
        code, message = (
            err_msg.split(": ", 1) if ": " in err_msg else ("FORMAT_ERROR", err_msg)
        )
        return line_no, None, {"line": line_no, "code": code, "message": message}


def validate_lines(
    line_generator: Generator[Tuple[int, str], None, None],
) -> Generator[
    Tuple[int, Submission, None]
    | Tuple[int, None, Dict[str, Any]],
    None,
    None,
]:
    """Generator: validate and parse JSON lines from the read pipeline."""
    for line_no, raw_line in line_generator:
        yield parse_submission(line_no, raw_line)


def deduplicate(
    validated_generator: Generator[
        Tuple[int, Submission, None]
        | Tuple[int, None, Dict[str, Any]],
        None,
        None,
    ],
) -> Generator[Tuple[str, Submission], None, None]:
    """Generator: yield unique submissions, skipping duplicates."""
    seen: Dict[str, str] = {}
    for result in validated_generator:
        line_no, submission, error = result
        if error is not None:
            # Error from upstream — skip
            continue
        sub_id = submission.submission_id
        code_hash = _compute_hash(submission.code)

        if sub_id in seen:
            # Duplicate or ID conflict — skip
            continue
        seen[sub_id] = code_hash
        yield sub_id, submission


class IngestionService:
    """Processes stream, validates schema, and resolves ID collisions.

    Uses a lazy generator pipeline (read → validate → deduplicate)
    to process data chunk-by-chunk and keep memory usage low.
    """

    def __init__(self) -> None:
        self.repository: Dict[str, Tuple[Submission, str]] = {}
        self.accepted: List[Submission] = []
        self.duplicates: List[str] = []
        self.errors: List[Dict[str, Any]] = []
        self.assignment: Assignment = Assignment(assignment_id="demo")

    def process_file(self, filepath: str) -> None:
        """Process a JSONL file using the lazy generator pipeline.

        Pipeline: read_lines → validate_lines → deduplicate
        Each stage is a generator, processing data chunk-by-chunk.
        """
        permitted_templates: List[str] = []
        with open(filepath, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line:
                first_data = json.loads(first_line)
                if "permitted_templates" in first_data:
                    permitted_templates = first_data["permitted_templates"]

        self.assignment = Assignment(
            assignment_id="demo",
            permitted_templates=permitted_templates,
        )

        seen_ids: Dict[str, str] = {}
        lines = read_lines(filepath)
        validated = validate_lines(lines)

        for result in validated:
            line_no, submission, error = result

            if error is not None:
                self.errors.append(error)
                continue

            sub_id = submission.submission_id
            code_hash = _compute_hash(submission.code)

            if sub_id in seen_ids:
                existing_hash = seen_ids[sub_id]
                if existing_hash != code_hash:
                    self.errors.append(
                        {
                            "line": line_no,
                            "code": "ID_CONFLICT",
                            "submission_id": sub_id,
                            "message": "Hash mismatch for existing ID",
                        }
                    )
                else:
                    self.duplicates.append(sub_id)
                continue

            seen_ids[sub_id] = code_hash
            self.repository[sub_id] = (submission, code_hash)
            self.accepted.append(submission)
