from typing import Callable, Dict, List, Protocol

from educode.models import AnalysisReport, Assignment, Submission


class Analyzer(Protocol):
    """Core interface for all extensible analyzers."""

    def analyze(
        self, submission: Submission, context: Assignment
    ) -> AnalysisReport: ...


class AnalyzerRegistry:
    """Explicit Factory Registry for internal analyzers."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], Analyzer]] = {}

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
