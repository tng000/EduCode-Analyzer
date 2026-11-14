from educode.ast_parser import ASTParser, parse_code


class TestParseCode:
    def test_valid_function(self):
        code = "def foo():\n    return 42\n"
        report = parse_code(code)
        assert report.has_syntax_error is False
        assert report.num_functions == 1
        assert report.num_nodes > 0
        assert report.tree is not None

    def test_valid_class(self):
        code = "class MyClass:\n    pass\n"
        report = parse_code(code)
        assert report.has_syntax_error is False
        assert report.num_classes == 1

    def test_valid_with_loop(self):
        code = "for i in range(10):\n    print(i)\n"
        report = parse_code(code)
        assert report.has_syntax_error is False
        assert report.num_loops == 1

    def test_valid_with_assignment(self):
        code = "x = 1\ny = 2\n"
        report = parse_code(code)
        assert report.has_syntax_error is False
        assert report.num_assignments == 2

    def test_valid_with_return(self):
        code = "def foo():\n    return 1\n    return 2\n"
        report = parse_code(code)
        assert report.has_syntax_error is False
        assert report.num_returns == 2

    def test_valid_with_call(self):
        code = "print('hello')\nlen([1, 2, 3])\n"
        report = parse_code(code)
        assert report.has_syntax_error is False
        assert report.num_calls == 2

    def test_empty_code(self):
        report = parse_code("")
        assert report.has_syntax_error is False
        assert report.num_nodes == 0

    def test_whitespace_only_code(self):
        report = parse_code("   \n  \n  ")
        assert report.has_syntax_error is False
        assert report.num_nodes == 0

    def test_syntax_error_missing_colon(self):
        code = "def foo()\n    pass\n"
        report = parse_code(code)
        assert report.has_syntax_error is True
        assert report.error is not None
        assert report.error.code == "SYNTAX_ERROR"
        assert "Line" in report.error.message

    def test_syntax_error_invalid_syntax(self):
        code = "if x = 5:\n    pass\n"
        report = parse_code(code)
        assert report.has_syntax_error is True
        assert report.error is not None
        assert report.error.code == "SYNTAX_ERROR"

    def test_syntax_error_incomplete_code(self):
        code = "def foo("
        report = parse_code(code)
        assert report.has_syntax_error is True
        assert report.error is not None

    def test_node_types_list(self):
        code = "def foo():\n    x = 1\n    return x\n"
        report = parse_code(code)
        assert "FunctionDef" in report.node_types
        assert "Assign" in report.node_types
        assert "Return" in report.node_types


class TestASTParser:
    def setup_method(self):
        self.parser = ASTParser()

    def test_parse_valid_code(self):
        code = "def foo():\n    return 42\n"
        report = self.parser.parse(code)
        assert report.has_syntax_error is False
        assert report.num_functions == 1

    def test_parse_caching(self):
        code = "def foo():\n    return 42\n"
        report1 = self.parser.parse(code)
        report2 = self.parser.parse(code)
        assert report1 is report2  # Same cached object

    def test_parse_syntax_error(self):
        code = "def foo("
        report = self.parser.parse(code)
        assert report.has_syntax_error is True
        assert report.error is not None

    def test_structural_signature_valid(self):
        code = "def foo():\n    for i in range(10):\n        print(i)\n"
        sig = self.parser.get_structural_signature(code)
        assert "SYNTAX_ERROR" not in sig
        assert "FunctionDef" in sig
        assert "For" in sig
        assert "Call" in sig

    def test_structural_signature_error(self):
        code = "def foo("
        sig = self.parser.get_structural_signature(code)
        assert sig == ["SYNTAX_ERROR"]

    def test_structural_similarity_identical(self):
        code = "def foo():\n    return 42\n"
        sim = self.parser.structural_similarity(code, code)
        assert sim is not None
        assert sim == 1.0

    def test_structural_similarity_different(self):
        code1 = "def foo():\n    return 42\n"
        code2 = "class Bar:\n    pass\n"
        sim = self.parser.structural_similarity(code1, code2)
        assert sim is not None
        assert sim < 1.0

    def test_structural_similarity_empty(self):
        sim = self.parser.structural_similarity("", "")
        assert sim is None

    def test_structural_similarity_one_empty(self):
        code = "def foo():\n    return 42\n"
        sim = self.parser.structural_similarity(code, "")
        assert sim is not None
        assert sim < 1.0
