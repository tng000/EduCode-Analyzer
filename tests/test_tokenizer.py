from educode.tokenizer import CodeTokenizer


class TestCodeTokenizer:
    def setup_method(self):
        self.tokenizer = CodeTokenizer()

    def test_simple_function(self):
        code = "def foo():\n    return 42\n"
        tokens = self.tokenizer.tokenize(code)
        assert "def" in tokens
        assert "foo" in tokens
        assert "return" in tokens
        assert "42" in tokens

    def test_strips_comments(self):
        code = "x = 1  # this is a comment\n"
        tokens = self.tokenizer.tokenize(code)
        assert "this" not in tokens
        assert "comment" not in tokens

    def test_strips_docstrings(self):
        code = '"""\nThis is a docstring.\n"""\ndef foo():\n    pass\n'
        tokens = self.tokenizer.tokenize(code)
        assert "docstring" not in tokens
        assert "def" in tokens

    def test_multiple_functions(self):
        code = "def foo():\n    pass\n\ndef bar():\n    pass\n"
        tokens = self.tokenizer.tokenize(code)
        assert "foo" in tokens
        assert "bar" in tokens

    def test_empty_code(self):
        code = ""
        tokens = self.tokenizer.tokenize(code)
        assert tokens == []
