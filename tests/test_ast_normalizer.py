from educode.ast_normalizer import normalize_code


class TestNormalizeCode:
    def test_simple_variable(self):
        code = "x = 1\n"
        result = normalize_code(code)
        assert "var_" in result
        assert "x = 1" not in result

    def test_multiple_variables(self):
        code = "x = 1\ny = 2\nz = 3\n"
        result = normalize_code(code)
        # All variables should be renamed
        assert "var_1" in result
        assert "var_2" in result
        assert "var_3" in result
        # Original names should not appear
        assert "x = " not in result
        assert "y = " not in result
        assert "z = " not in result

    def test_function_parameter(self):
        code = "def foo(n):\n    return n\n"
        result = normalize_code(code)
        assert "var_" in result
        # Function name should be normalized
        assert "def var_" in result

    def test_function_body_variables(self):
        code = "def calculate(x):\n    result = x * 2\n    return result\n"
        result = normalize_code(code)
        assert "var_" in result

    def test_for_loop_variable(self):
        code = "for i in range(10):\n    print(i)\n"
        result = normalize_code(code)
        assert "var_" in result

    def test_for_loop_iter_variable(self):
        """Variables used in range() iterator should also be normalized."""
        code = "for i in range(n):\n    print(i)\n"
        result = normalize_code(code)
        assert "var_" in result
        assert "range(n)" not in result  # n should be normalized too
        assert "range(var_" in result

    def test_identical_code_normalized_same(self):
        code = "x = 1\ny = 2\n"
        result1 = normalize_code(code)
        result2 = normalize_code(code)
        assert result1 == result2

    def test_different_structure_different_normalization(self):
        code1 = "x = 1\ny = 2\n"
        code2 = "a = 1\nb = 2\nc = 3\n"
        result1 = normalize_code(code1)
        result2 = normalize_code(code2)
        # Different number of variables -> different normalized output
        assert result1 != result2

    def test_syntax_error_returns_original(self):
        code = "def foo(\n"
        result = normalize_code(code)
        assert result == code

    def test_empty_code(self):
        result = normalize_code("")
        assert result == ""

    def test_whitespace_only(self):
        code = "   \n  \n"
        result = normalize_code(code)
        assert result == code

    def test_complex_function(self):
        code = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        result = normalize_code(code)
        assert "var_" in result
        # Result should be valid Python (no syntax errors)
        # (If it had syntax errors, it would still be a string)

    def test_class_normalization(self):
        code = "class MyClass:\n    def __init__(self):\n        self.value = 42\n"
        result = normalize_code(code)
        assert "var_" in result

    def test_variable_order_preserved(self):
        """Variables should be normalized in order of first appearance."""
        code = "first = 1\nsecond = 2\nthird = 3\n"
        result = normalize_code(code)
        # var_1 should appear before var_2, which should appear before var_3
        pos_1 = result.index("var_1")
        pos_2 = result.index("var_2")
        pos_3 = result.index("var_3")
        assert pos_1 < pos_2 < pos_3

    def test_list_comprehension(self):
        code = "result = [i * 2 for i in range(10)]\n"
        result = normalize_code(code)
        assert "var_" in result

    def test_multiple_functions(self):
        code = "def foo():\n    pass\n\n" "def bar():\n    pass\n"
        result = normalize_code(code)
        # Both functions should be normalized
        assert result.count("def var_") == 2

    def test_assignment_with_expression(self):
        code = "x = y + z * 2\n"
        result = normalize_code(code)
        assert "var_" in result

    def test_augmented_assignment(self):
        code = "x = 1\nx += 2\n"
        result = normalize_code(code)
        assert "var_" in result

    def test_nested_function(self):
        code = (
            "def outer():\n"
            "    def inner():\n"
            "        return 42\n"
            "    return inner()\n"
        )
        result = normalize_code(code)
        assert result.count("def var_") == 2
