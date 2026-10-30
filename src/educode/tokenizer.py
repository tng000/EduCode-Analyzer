"""Code tokenizer using Python's built-in tokenize library."""

import tokenize
from io import BytesIO
from typing import Iterator, List


class CodeTokenizer:
    """Tokenize Python source code, stripping comments, docstrings, and whitespace."""

    def tokenize(self, code: str) -> List[str]:
        """Return a clean sequence of meaningful code tokens.

        Strips comments, docstrings, and redundant whitespaces.
        """
        return list(self._token_generator(code))

    def _token_generator(self, code: str) -> Iterator[str]:
        """Yield meaningful tokens, skipping comments, docstrings, and whitespace."""
        try:
            tokens = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)
        except tokenize.TokenError:
            # If tokenization fails (e.g., incomplete code), fall back to simple split
            yield from code.split()
            return

        for tok in tokens:
            # Skip NEWLINE, INDENT, DEDENT, NL, ENCODING
            if tok.type in (
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENCODING,
            ):
                continue
            # Skip comments
            if tok.type == tokenize.COMMENT:
                continue
            # Skip docstrings (STRING tokens)
            if tok.type == tokenize.STRING:
                continue
            # Skip token string representation
            if tok.string.strip():
                yield tok.string
