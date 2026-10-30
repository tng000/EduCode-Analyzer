from typing import Callable, Dict, List, Optional, Protocol, Set, Tuple

from educode.models import AnalysisReport, Assignment, Submission
from educode.tokenizer import CodeTokenizer


class Analyzer(Protocol):
    """Core interface for all extensible analyzers."""

    def analyze(
        self, submission: Submission, context: Assignment
    ) -> AnalysisReport: ...


class LengthAnalyzer:
    """Counts non-empty lines in a submission's code."""

    def __init__(self) -> None:
        self.analyzer_id = "length_analyzer"
        self.analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: Assignment) -> AnalysisReport:
        non_empty_lines = sum(
            1 for line in submission.code.splitlines() if line.strip()
        )
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            score=non_empty_lines,
            rubric_feedback=f"Non-empty lines: {non_empty_lines}",
        )


class TemplateAnalyzer:
    """Finds exact template matches between submissions."""

    def __init__(self) -> None:
        self.analyzer_id = "template_analyzer"
        self.analyzer_version = "1.0.0"

    def analyze(self, submission: Submission, context: Assignment) -> AnalysisReport:
        evidence: List[Dict[str, object]] = []
        for template in context.permitted_templates:
            if template in submission.code:
                evidence.append(
                    {"type": "template_match", "template": template}
                )
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            similarity=1.0 if evidence else 0.0,
            rubric_feedback=f"Template matches found: {len(evidence)}",
            evidence=evidence,
        )


class JaccardAnalyzer:
    """Token-based Jaccard similarity using 3-grams."""

    def __init__(self) -> None:
        self.analyzer_id = "jaccard_analyzer"
        self.analyzer_version = "1.0.0"
        self._tokenizer = CodeTokenizer()

    def _get_ngrams(self, tokens: List[str], n: int = 3) -> Set[str]:
        """Generate n-grams from a list of tokens."""
        if len(tokens) < n:
            return set()
        return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}

    def analyze_pair(
        self, sub1: Submission, sub2: Submission
    ) -> Optional[float]:
        """Compute Jaccard similarity between two submissions.

        Returns None if the union of 3-grams is empty.
        """
        tokens1 = self._tokenizer.tokenize(sub1.code)
        tokens2 = self._tokenizer.tokenize(sub2.code)

        ngrams1 = self._get_ngrams(tokens1)
        ngrams2 = self._get_ngrams(tokens2)

        union = ngrams1 | ngrams2
        if len(union) == 0:
            return None

        intersection = ngrams1 & ngrams2
        return len(intersection) / len(union)

    def analyze(self, submission: Submission, context: Assignment) -> AnalysisReport:
        tokens = self._tokenizer.tokenize(submission.code)
        ngrams = self._get_ngrams(tokens)
        return AnalysisReport(
            analyzer_id=self.analyzer_id,
            analyzer_version=self.analyzer_version,
            score=len(ngrams),
            rubric_feedback=f"Token count: {len(tokens)}, 3-gram count: {len(ngrams)}",
        )


class AnalyzerRegistry:
    """Explicit Factory Registry for internal analyzers."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], Analyzer]] = {}
        self._register_builtins()

    def _register_builtins(self) -> None:
        """Auto-register all built-in analyzers."""
        self.register("length", LengthAnalyzer)
        self.register("template", TemplateAnalyzer)
        self.register("jaccard", JaccardAnalyzer)

    def register(self, name: str, factory: Callable[[], Analyzer]) -> None:
        """Register a factory function that creates an Analyzer instance."""
        self._factories[name] = factory

    def get_analyzer(self, name: str) -> Analyzer:
        """Instantiate and return an analyzer by its registered name."""
        if name not in self._factories:
            raise ValueError(f"Analyzer '{name}' is not registered.")
        return self._factories[name]()

    def list_analyzers(self) -> List[str]:
        return list(self._factories.keys())


def compute_pairwise_jaccard(
    submissions: List[Submission], top_k: int = 10
) -> List[Tuple[str, str, float]]:
    """Compute pairwise Jaccard similarity and return Top-K suspicious pairs.

    Returns a list of (student_id_1, student_id_2, similarity) tuples.
    Only considers pairs of different students. For each pair of students,
    takes the maximum similarity across all their submission pairs.
    """
    analyzer = JaccardAnalyzer()
    pair_max_sim: Dict[Tuple[str, str], float] = {}

    for i in range(len(submissions)):
        for j in range(i + 1, len(submissions)):
            sid1 = submissions[i].student_id
            sid2 = submissions[j].student_id

            # Skip if same student
            if sid1 == sid2:
                continue

            sim = analyzer.analyze_pair(submissions[i], submissions[j])
            if sim is not None and sim > 0.0:
                # Normalize pair order (alphabetical)
                pair_key = tuple(sorted((sid1, sid2)))
                if pair_key not in pair_max_sim or sim > pair_max_sim[pair_key]:
                    pair_max_sim[pair_key] = sim

    pairs = [
        (sid1, sid2, round(sim, 4))
        for (sid1, sid2), sim in pair_max_sim.items()
    ]

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]
