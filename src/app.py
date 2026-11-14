import dataclasses
import tempfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

from educode.analyzer import (
    AnalyzerRegistry,
    compute_pairwise_jaccard,
)
from educode.ast_normalizer import normalize_code
from educode.ast_parser import ASTParser
from educode.ingestion import IngestionService
from educode.models import Submission

ALGORITHM_TOKEN = "Token Baseline"
ALGORITHM_AST = "AST Structural"


def _run_token_analysis(
    service: IngestionService, registry: AnalyzerRegistry
) -> List[Dict[str, object]]:
    """Run token-based analysis (Jaccard + baseline analyzers)."""
    available = registry.list_analyzers()
    rows = []
    for sub in service.accepted:
        for analyzer_name in available:
            analyzer = registry.get_analyzer(analyzer_name)
            report = analyzer.analyze(sub, service.assignment)
            rows.append(
                {
                    "Submission ID": sub.submission_id,
                    "Student ID": sub.student_id,
                    "Analyzer": analyzer_name,
                    "Score": report.score,
                    "Similarity": report.similarity,
                    "Feedback": report.rubric_feedback,
                }
            )
    return rows


def _run_ast_analysis(
    service: IngestionService,
) -> List[Dict[str, object]]:
    """Run AST-based structural analysis on accepted submissions."""
    parser = ASTParser()
    rows = []
    for sub in service.accepted:
        report = parser.parse(sub.code)
        rows.append(
            {
                "Submission ID": sub.submission_id,
                "Student ID": sub.student_id,
                "Analyzer": "ast_parser",
                "Score": report.num_nodes,
                "Similarity": None,
                "Feedback": (
                    f"Functions: {report.num_functions}, "
                    f"Classes: {report.num_classes}, "
                    f"Assignments: {report.num_assignments}, "
                    f"Returns: {report.num_returns}, "
                    f"Loops: {report.num_loops}, "
                    f"Calls: {report.num_calls}"
                ),
            }
        )
        if report.has_syntax_error and report.error:
            rows.append(
                {
                    "Submission ID": sub.submission_id,
                    "Student ID": sub.student_id,
                    "Analyzer": "ast_parser_error",
                    "Score": None,
                    "Similarity": None,
                    "Feedback": f"{report.error.code}: {report.error.message}",
                }
            )

    # Add normalization results
    for sub in service.accepted:
        normalized = normalize_code(sub.code)
        # Truncate for display
        display = normalized[:200]
        if len(normalized) > 200:
            display += "..."
        rows.append(
            {
                "Submission ID": sub.submission_id,
                "Student ID": sub.student_id,
                "Analyzer": "ast_normalizer",
                "Score": len(normalized),
                "Similarity": None,
                "Feedback": display,
            }
        )

    return rows


def _run_ast_pairwise_analysis(
    submissions: List[Submission], top_k: int
) -> List[tuple]:
    """Run AST-based pairwise structural similarity analysis."""
    parser = ASTParser()
    pair_max_sim: Dict[tuple[str, str], float] = {}

    for i in range(len(submissions)):
        for j in range(i + 1, len(submissions)):
            sid1 = submissions[i].student_id
            sid2 = submissions[j].student_id

            if sid1 == sid2:
                continue

            sim = parser.structural_similarity(submissions[i].code, submissions[j].code)
            if sim is not None and sim > 0.0:
                pair_key = tuple(sorted((sid1, sid2)))
                if pair_key not in pair_max_sim or sim > pair_max_sim[pair_key]:
                    pair_max_sim[pair_key] = sim

    pairs = [(sid1, sid2, round(sim, 4)) for (sid1, sid2), sim in pair_max_sim.items()]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


def main() -> None:
    """Run the Streamlit web application."""
    st.set_page_config(page_title="EduCode Analyzer", layout="wide")
    st.title("EduCode Analyzer")

    # Algorithm selection in sidebar
    st.sidebar.header("Analysis Engine")
    algorithm = st.sidebar.radio(
        "Select algorithm",
        [ALGORITHM_TOKEN, ALGORITHM_AST],
        index=0,
    )

    uploaded_file = st.file_uploader("Upload submissions", type=["jsonl"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            service = IngestionService()
            service.process_file(tmp_path)

            accepted_list = []
            for sub in service.accepted:
                sub_dict = dataclasses.asdict(sub)
                sub_dict["timestamp"] = sub.timestamp.isoformat()
                accepted_list.append(sub_dict)

            output = {
                "accepted": accepted_list,
                "duplicate": service.duplicates,
                "errors": service.errors,
            }

            st.subheader("Processing Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accepted", len(output["accepted"]))
            col2.metric("Duplicates", len(output["duplicate"]))
            col3.metric("Errors", len(output["errors"]))

            # Display selected algorithm
            st.info(f"Active engine: **{algorithm}**")

            if service.accepted:
                st.subheader("Analysis Results")

                if algorithm == ALGORITHM_TOKEN:
                    registry = AnalyzerRegistry()
                    available = registry.list_analyzers()
                    st.subheader(f"Analyzers ({len(available)})")
                    st.markdown(f"Available: {', '.join(available)}")

                    rows = _run_token_analysis(service, registry)
                else:
                    rows = _run_ast_analysis(service)

                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, width="stretch")

            # Plagiarism Detection Dashboard
            if len(service.accepted) >= 2:
                st.subheader("Plagiarism Detection Results")
                top_k = st.slider("Top-K suspicious pairs", 2, 20, 10)

                if algorithm == ALGORITHM_TOKEN:
                    suspicious_pairs = compute_pairwise_jaccard(
                        service.accepted, top_k=top_k
                    )
                else:
                    suspicious_pairs = _run_ast_pairwise_analysis(
                        service.accepted, top_k
                    )

                # Build student_id -> submission_id mapping for tooltips
                student_submissions: Dict[str, List[str]] = {}
                for sub in service.accepted:
                    student_submissions.setdefault(sub.student_id, []).append(
                        sub.submission_id
                    )

                if suspicious_pairs:
                    df = pd.DataFrame(
                        suspicious_pairs,
                        columns=["Student A", "Student B", "Similarity Score"],
                    )
                    st.dataframe(df, width="stretch")

                    # Highlight high-similarity pairs
                    high_sim = [p for p in suspicious_pairs if p[2] >= 0.8]
                    if high_sim:
                        st.warning(
                            f"Found {len(high_sim)} pair(s) " f"with similarity >= 80%"
                        )
                        for student_a, student_b, score in high_sim:
                            subs_a = ", ".join(student_submissions.get(student_a, []))
                            subs_b = ", ".join(student_submissions.get(student_b, []))
                            st.info(
                                f"**{student_a}** (submissions: {subs_a}) "
                                f"↔ "
                                f"**{student_b}** (submissions: {subs_b}): "
                                f"{score:.1%}"
                            )
                else:
                    st.info("No suspicious pairs detected.")

            st.json(output)

        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
