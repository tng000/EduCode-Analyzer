import types
from pathlib import Path
from typing import Iterator, Optional


class JSONLSource:
    """Context manager for safe reading of JSONL files."""

    def __init__(self, filepath: Path | str) -> None:
        self.filepath = Path(filepath)
        self._file = None

    def __enter__(self) -> "JSONLSource":
        self._file = self.filepath.open("r", encoding="utf-8")
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[types.TracebackType],
    ) -> None:
        if self._file is not None:
            self._file.close()

    def __iter__(self) -> Iterator[tuple[int, str]]:
        if self._file is None:
            raise RuntimeError("Source must be used as a context manager.")
        for line_no, line in enumerate(self._file, start=1):
            if line.strip():
                yield line_no, line
