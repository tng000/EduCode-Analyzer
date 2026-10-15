import dataclasses
import tempfile
from pathlib import Path

import streamlit as st

from educode.ingestion import IngestionService


def main() -> None:
    """Run the Streamlit web application."""
    st.set_page_config(page_title="EduCode Analyzer", layout="wide")
    st.title("EduCode Analyzer MVP")
    st.markdown(
        "Upload student submissions (.jsonl) to validate and store them in-memory."
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

            st.json(output)

        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
