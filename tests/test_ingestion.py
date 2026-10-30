import json
import tempfile
from pathlib import Path

from educode.ingestion import (
    IngestionService,
    deduplicate,
    parse_submission,
    validate_and_parse,
)
from educode.models import AIDeclaration, Submission


class TestValidateAndParse:
    def test_valid_submission(self):
        data = {
            "submission_id": "1",
            "student_id": "s1",
            "timestamp": "2024-01-01T00:00:00Z",
            "code": "print('hello')",
            "ai_declaration": {"used": False},
        }
        submission = validate_and_parse(json.dumps(data))
        assert submission.submission_id == "1"
        assert submission.student_id == "s1"
        assert submission.code == "print('hello')"

    def test_missing_fields(self):
        data = {"submission_id": "1"}
        try:
            validate_and_parse(json.dumps(data))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Missing required fields" in str(e)

    def test_invalid_timestamp(self):
        data = {
            "submission_id": "1",
            "student_id": "s1",
            "timestamp": "invalid",
            "code": "x = 1",
            "ai_declaration": {"used": False},
        }
        try:
            validate_and_parse(json.dumps(data))
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Invalid timestamp" in str(e)


class TestParseSubmission:
    def test_success(self):
        data = {
            "submission_id": "1",
            "student_id": "s1",
            "timestamp": "2024-01-01T00:00:00Z",
            "code": "print('hello')",
            "ai_declaration": {"used": False},
        }
        line_no, submission, error = parse_submission(1, json.dumps(data))
        assert error is None
        assert submission is not None
        assert submission.submission_id == "1"

    def test_parse_failure(self):
        line_no, submission, error = parse_submission(1, "not valid json")
        assert error is not None
        assert submission is None
        assert error["line"] == 1
        assert error["code"] == "FORMAT_ERROR"


class TestDeduplicate:
    def test_removes_duplicates(self):
        def gen():
            yield (1, Submission(
                submission_id="1",
                student_id="s1",
                timestamp="2024-01-01T00:00:00Z",
                code="x = 1",
                ai_declaration=AIDeclaration(used=False),
            ), None)
            yield (2, Submission(
                submission_id="1",
                student_id="s1",
                timestamp="2024-01-01T00:00:01Z",
                code="x = 1",
                ai_declaration=AIDeclaration(used=False),
            ), None)
            yield (3, Submission(
                submission_id="3",
                student_id="s2",
                timestamp="2024-01-01T00:00:02Z",
                code="y = 2",
                ai_declaration=AIDeclaration(used=False),
            ), None)

        unique = list(deduplicate(gen()))
        assert len(unique) == 2


class TestIngestionService:
    def test_permitted_templates_extracted(self):
        factorial_code = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)"
        )
        fibonacci_code = (
            "def fibonacci(n):\n"
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a"
        )
        data = [
            {
                "submission_id": "1",
                "student_id": "s1",
                "timestamp": "2024-01-01T00:00:00Z",
                "code": factorial_code,
                "ai_declaration": {"used": False},
                "permitted_templates": [
                    "def factorial(n):",
                    "def fibonacci(n):",
                ],
            },
            {
                "submission_id": "2",
                "student_id": "s2",
                "timestamp": "2024-01-01T00:01:00Z",
                "code": fibonacci_code,
                "ai_declaration": {"used": False},
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
            tmp_path = f.name

        try:
            service = IngestionService()
            service.process_file(tmp_path)

            assert len(service.accepted) == 2
            assert service.assignment.permitted_templates == [
                "def factorial(n):",
                "def fibonacci(n):",
            ]
        finally:
            Path(tmp_path).unlink()
