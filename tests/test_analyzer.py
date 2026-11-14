from educode.analyzer import (
    JaccardAnalyzer,
    LengthAnalyzer,
    TemplateAnalyzer,
    compute_pairwise_jaccard,
)
from educode.models import AIDeclaration, Assignment, Submission


class TestLengthAnalyzer:
    def setup_method(self):
        self.analyzer = LengthAnalyzer()

    def test_simple_code(self):
        submission = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="x = 1\ny = 2\nz = 3\n",
            ai_declaration=AIDeclaration(used=False),
        )
        report = self.analyzer.analyze(submission, Assignment(assignment_id="a1"))
        assert report.score == 3
        assert "Non-empty lines: 3" in report.rubric_feedback

    def test_empty_code(self):
        submission = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="",
            ai_declaration=AIDeclaration(used=False),
        )
        report = self.analyzer.analyze(submission, Assignment(assignment_id="a1"))
        assert report.score == 0

    def test_strips_blank_lines(self):
        submission = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="x = 1\n\n\ny = 2\n",
            ai_declaration=AIDeclaration(used=False),
        )
        report = self.analyzer.analyze(submission, Assignment(assignment_id="a1"))
        assert report.score == 2


class TestTemplateAnalyzer:
    def setup_method(self):
        self.analyzer = TemplateAnalyzer()

    def test_no_template_match(self):
        submission = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="x = 1\n",
            ai_declaration=AIDeclaration(used=False),
        )
        context = Assignment(assignment_id="a1", permitted_templates=["template_a"])
        report = self.analyzer.analyze(submission, context)
        assert report.similarity == 0.0
        assert len(report.evidence) == 0

    def test_template_match(self):
        submission = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="template_a\nx = 1\n",
            ai_declaration=AIDeclaration(used=False),
        )
        context = Assignment(assignment_id="a1", permitted_templates=["template_a"])
        report = self.analyzer.analyze(submission, context)
        assert report.similarity == 1.0
        assert len(report.evidence) == 1
        assert report.evidence[0]["type"] == "template_match"


class TestJaccardAnalyzer:
    def setup_method(self):
        self.analyzer = JaccardAnalyzer()

    def test_identical_code(self):
        submission = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="def foo():\n    return 42\n",
            ai_declaration=AIDeclaration(used=False),
        )
        sim = self.analyzer.analyze_pair(submission, submission)
        assert sim == 1.0

    def test_different_code(self):
        sub1 = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="def foo():\n    return 42\n",
            ai_declaration=AIDeclaration(used=False),
        )
        sub2 = Submission(
            submission_id="2",
            student_id="s2",
            timestamp="2024-01-01T00:00:00Z",
            code="def bar():\n    return 99\n",
            ai_declaration=AIDeclaration(used=False),
        )
        sim = self.analyzer.analyze_pair(sub1, sub2)
        assert sim is not None
        assert sim < 1.0

    def test_empty_code(self):
        sub1 = Submission(
            submission_id="1",
            student_id="s1",
            timestamp="2024-01-01T00:00:00Z",
            code="",
            ai_declaration=AIDeclaration(used=False),
        )
        sub2 = Submission(
            submission_id="2",
            student_id="s2",
            timestamp="2024-01-01T00:00:00Z",
            code="",
            ai_declaration=AIDeclaration(used=False),
        )
        sim = self.analyzer.analyze_pair(sub1, sub2)
        assert sim is None


class TestComputePairwiseJaccard:
    def test_no_duplicates(self):
        submissions = [
            Submission(
                submission_id="1",
                student_id="s1",
                timestamp="2024-01-01T00:00:00Z",
                code="def foo():\n    x = 1\n    return x\n",
                ai_declaration=AIDeclaration(used=False),
            ),
            Submission(
                submission_id="2",
                student_id="s2",
                timestamp="2024-01-01T00:00:00Z",
                code="def bar():\n    x = 1\n    return x\n",
                ai_declaration=AIDeclaration(used=False),
            ),
        ]
        pairs = compute_pairwise_jaccard(submissions, top_k=10)
        assert len(pairs) == 1
        assert pairs[0][0] == "s1"
        assert pairs[0][1] == "s2"

    def test_top_k_limit(self):
        submissions = [
            Submission(
                submission_id=str(i),
                student_id=f"s{i}",
                timestamp="2024-01-01T00:00:00Z",
                code=f"x = {i}\n",
                ai_declaration=AIDeclaration(used=False),
            )
            for i in range(10)
        ]
        pairs = compute_pairwise_jaccard(submissions, top_k=5)
        assert len(pairs) <= 5

    def test_same_student_not_included(self):
        """Pairs from the same student should be excluded."""
        submissions = [
            Submission(
                submission_id="1",
                student_id="s1",
                timestamp="2024-01-01T00:00:00Z",
                code="def foo():\n    return 1\n",
                ai_declaration=AIDeclaration(used=False),
            ),
            Submission(
                submission_id="2",
                student_id="s1",
                timestamp="2024-01-01T00:01:00Z",
                code="def foo():\n    return 1\n",
                ai_declaration=AIDeclaration(used=False),
            ),
            Submission(
                submission_id="3",
                student_id="s2",
                timestamp="2024-01-01T00:02:00Z",
                code="def bar():\n    return 2\n",
                ai_declaration=AIDeclaration(used=False),
            ),
        ]
        pairs = compute_pairwise_jaccard(submissions, top_k=10)
        # Only s1-s2 pair should appear, not s1-s1
        assert len(pairs) == 1
        pair_students = {pairs[0][0], pairs[0][1]}
        assert pair_students == {"s1", "s2"}

    def test_no_duplicate_pairs(self):
        """Each student pair should appear only once (alphabetically ordered)."""
        submissions = [
            Submission(
                submission_id="1",
                student_id="s1",
                timestamp="2024-01-01T00:00:00Z",
                code="def foo():\n    return 1\n",
                ai_declaration=AIDeclaration(used=False),
            ),
            Submission(
                submission_id="2",
                student_id="s2",
                timestamp="2024-01-01T00:01:00Z",
                code="def foo():\n    return 1\n",
                ai_declaration=AIDeclaration(used=False),
            ),
            Submission(
                submission_id="3",
                student_id="s3",
                timestamp="2024-01-01T00:02:00Z",
                code="def bar():\n    return 2\n",
                ai_declaration=AIDeclaration(used=False),
            ),
        ]
        pairs = compute_pairwise_jaccard(submissions, top_k=10)
        # Should have s1-s2 and s1-s3, but not s2-s1 or s3-s1
        student_ids_in_pairs = set()
        for sid1, sid2, _ in pairs:
            pair_key = tuple(sorted((sid1, sid2)))
            assert pair_key not in student_ids_in_pairs
            student_ids_in_pairs.add(pair_key)

    def test_max_similarity_kept(self):
        """When same student pair has multiple submissions, max similarity is kept."""
        submissions = [
            Submission(
                submission_id="1",
                student_id="s1",
                timestamp="2024-01-01T00:00:00Z",
                code="def foo():\n    return 1\n",
                ai_declaration=AIDeclaration(used=False),
            ),
            Submission(
                submission_id="2",
                student_id="s1",
                timestamp="2024-01-01T00:01:00Z",
                code="def foo():\n    return 1\n",
                ai_declaration=AIDeclaration(used=False),
            ),
            Submission(
                submission_id="3",
                student_id="s2",
                timestamp="2024-01-01T00:02:00Z",
                code="def foo():\n    return 1\n",
                ai_declaration=AIDeclaration(used=False),
            ),
        ]
        pairs = compute_pairwise_jaccard(submissions, top_k=10)
        assert len(pairs) == 1
        assert pairs[0][2] == 1.0  # identical code -> 100% similarity
