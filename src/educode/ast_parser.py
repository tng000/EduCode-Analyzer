"""AST-based code analysis module using Python's ast module."""

import ast
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ASTError:
    """Represents a parse error with a readable status message."""

    code: str
    message: str


@dataclass
class ASTReport:
    """Analysis report from AST-based structural analysis."""

    has_syntax_error: bool = False
    error: Optional[ASTError] = None
    num_nodes: int = 0
    node_types: List[str] = field(default_factory=list)
    num_functions: int = 0
    num_classes: int = 0
    num_assignments: int = 0
    num_returns: int = 0
    num_loops: int = 0
    num_calls: int = 0
    tree: Optional[ast.AST] = None


def _count_nodes(tree: ast.AST) -> tuple:
    """Traverse the AST and count node types.

    Returns a tuple of (total_count, type_counts, functions, classes,
    assignments, returns, loops, calls).
    """
    type_counts: Dict[str, int] = {}
    num_functions = 0
    num_classes = 0
    num_assignments = 0
    num_returns = 0
    num_loops = 0
    num_calls = 0

    for node in ast.walk(tree):
        node_type = type(node).__name__
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            num_functions += 1
        elif isinstance(node, ast.ClassDef):
            num_classes += 1
        elif isinstance(node, ast.Assign):
            num_assignments += 1
        elif isinstance(node, ast.Return):
            num_returns += 1
        elif isinstance(node, (ast.For, ast.While)):
            num_loops += 1
        elif isinstance(node, ast.Call):
            num_calls += 1

    total = sum(type_counts.values())
    return (
        total,
        type_counts,
        num_functions,
        num_classes,
        num_assignments,
        num_returns,
        num_loops,
        num_calls,
    )


def parse_code(code: str) -> ASTReport:
    """Parse Python source code into an AST and return an analysis report.

    Handles SyntaxError gracefully by returning a readable error status
    instead of stopping the pipeline.

    Args:
        code: Python source code string to parse.

    Returns:
        ASTReport with parsed tree and structural metrics, or error status.
    """
    if not code or not code.strip():
        return ASTReport(
            has_syntax_error=False,
            num_nodes=0,
            node_types=[],
            num_functions=0,
            num_classes=0,
            num_assignments=0,
            num_returns=0,
            num_loops=0,
            num_calls=0,
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        line_no = e.lineno or "?"
        msg = str(e.msg) or "invalid syntax"
        return ASTReport(
            has_syntax_error=True,
            error=ASTError(
                code="SYNTAX_ERROR",
                message=f"Line {line_no}: {msg}",
            ),
        )

    (
        num_nodes,
        type_counts,
        num_functions,
        num_classes,
        num_assignments,
        num_returns,
        num_loops,
        num_calls,
    ) = _count_nodes(tree)

    return ASTReport(
        has_syntax_error=False,
        tree=tree,
        num_nodes=num_nodes,
        node_types=sorted(type_counts.keys()),
        num_functions=num_functions,
        num_classes=num_classes,
        num_assignments=num_assignments,
        num_returns=num_returns,
        num_loops=num_loops,
        num_calls=num_calls,
    )


class ASTParser:
    """Provides AST-based structural analysis for student code.

    Parses code into Abstract Syntax Trees and extracts structural
    metrics for comparison and plagiarism detection.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, ASTReport] = {}

    def parse(self, code: str) -> ASTReport:
        """Parse code and return an ASTReport.

        Uses caching to avoid re-parsing identical code strings.
        """
        if code in self._cache:
            return self._cache[code]

        report = parse_code(code)
        self._cache[code] = report
        return report

    def get_structural_signature(self, code: str) -> List[str]:
        """Extract a structural signature from the code's AST.

        Returns a list of strings representing the sequence of significant
        node types encountered in the AST, which can be used for structural
        comparison between submissions.
        """
        report = self.parse(code)
        if report.has_syntax_error:
            return ["SYNTAX_ERROR"]
        if report.tree is None:
            return []

        signature: List[str] = []
        for node in ast.walk(report.tree):
            node_type = type(node).__name__
            # Focus on structural nodes
            if node_type in (
                "FunctionDef",
                "AsyncFunctionDef",
                "ClassDef",
                "Assign",
                "AugAssign",
                "Return",
                "For",
                "While",
                "If",
                "Call",
                "Expr",
                "BinOp",
                "Compare",
                "UnaryOp",
                "Assign",
                "AnnAssign",
                "Assert",
                "Raise",
                "Try",
                "With",
                "AsyncWith",
                "Global",
                "Nonlocal",
                "Delete",
            ):
                signature.append(node_type)

        return signature

    def structural_similarity(self, code1: str, code2: str) -> Optional[float]:
        """Compute structural similarity between two code snippets using AST signatures.

        Uses Jaccard similarity on the set of structural node types.
        Returns None if both signatures are empty.
        """
        sig1 = self.get_structural_signature(code1)
        sig2 = self.get_structural_signature(code2)

        if not sig1 and not sig2:
            return None

        set1 = set(sig1)
        set2 = set(sig2)
        union = set1 | set2

        if not union:
            return None

        intersection = set1 & set2
        return len(intersection) / len(union)
