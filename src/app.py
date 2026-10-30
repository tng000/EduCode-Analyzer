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
from educode.ingestion import IngestionService


def main() -> None:
    """Run the Streamlit web application."""
    st.set_page_config(page_title="EduCode Analyzer", layout="wide")
    st.title("EduCode Analyzer MVP")
    st.markdown(
        "Upload student submissions (.jsonl) to validate, store, and detect plagiarism."
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

            # Run analyzers via registry
            registry = AnalyzerRegistry()
            available = registry.list_analyzers()
            st.subheader(f"Analyzers ({len(available)})")
            st.markdown(f"Available: {', '.join(available)}")

            if service.accepted:
                st.subheader("Analysis Results")

                rows = []
                for sub in service.accepted:
                    for analyzer_name in available:
                        analyzer = registry.get_analyzer(analyzer_name)
                        report = analyzer.analyze(
                            sub, service.assignment
                        )
                        rows.append({
                            "Submission ID": sub.submission_id,
                            "Student ID": sub.student_id,
                            "Analyzer": analyzer_name,
                            "Score": report.score,
                            "Similarity": report.similarity,
                            "Feedback": report.rubric_feedback,
                        })

                if rows:
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)

            # Plagiarism Detection Dashboard
            if len(service.accepted) >= 2:
                st.subheader("Plagiarism Detection Results")
                top_k = st.slider("Top-K suspicious pairs", 2, 20, 10)
                suspicious_pairs = compute_pairwise_jaccard(
                    service.accepted, top_k=top_k
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
                    st.dataframe(df, use_container_width=True)

                    # Highlight high-similarity pairs
                    high_sim = [p for p in suspicious_pairs if p[2] >= 0.8]
                    if high_sim:
                        st.warning(
                            f"Found {len(high_sim)} pair(s) "
                            f"with similarity >= 80%"
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
