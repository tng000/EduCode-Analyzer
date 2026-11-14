"""AST node transformer for variable and function name normalization."""

import ast
from typing import Dict, List

from educode.ast_parser import parse_code


class _VariableNormalizer(ast.NodeTransformer):
    """AST NodeTransformer that renames variables sequentially.

    Traverses the AST and normalizes all variables and functions
    in the order of appearance to var_1, var_2, var_3, etc.
    Skips Python built-in functions (range, print, int, len, etc).
    """

    # Python built-in functions that should not be normalized
    _BUILTINS = frozenset(
        {
            "abs",
            "aiter",
            "all",
            "any",
            "anext",
            "ascii",
            "bin",
            "bool",
            "breakpoint",
            "bytearray",
            "bytes",
            "callable",
            "chr",
            "classmethod",
            "compile",
            "complex",
            "delattr",
            "dict",
            "dir",
            "divmod",
            "enumerate",
            "eval",
            "exec",
            "filter",
            "float",
            "format",
            "frozenset",
            "getattr",
            "globals",
            "hasattr",
            "hash",
            "help",
            "hex",
            "id",
            "input",
            "int",
            "isinstance",
            "issubclass",
            "iter",
            "len",
            "list",
            "locals",
            "map",
            "max",
            "memoryview",
            "min",
            "next",
            "object",
            "oct",
            "open",
            "ord",
            "pow",
            "print",
            "property",
            "range",
            "repr",
            "reversed",
            "round",
            "set",
            "setattr",
            "slice",
            "sorted",
            "staticmethod",
            "str",
            "sum",
            "super",
            "tuple",
            "type",
            "vars",
            "zip",
        }
    )

    def __init__(self) -> None:
        self._name_map: Dict[str, str] = {}
        self._counter: int = 0

    def _next_name(self) -> str:
        """Generate the next normalized name."""
        self._counter += 1
        return f"var_{self._counter}"

    def _get_or_create_name(self, original: str) -> str:
        """Get normalized name for a variable, creating one if needed.

        Skips Python built-in functions — they are never normalized.
        """
        if original in self._BUILTINS:
            return original
        if original not in self._name_map:
            self._name_map[original] = self._next_name()
        return self._name_map[original]

    def _visit_name(self, node: ast.Name, ctx_type: type) -> ast.Name:
        """Normalize a Name node (variable reference).

        Skips Python built-in functions.
        """
        if node.id in self._BUILTINS:
            return node
        if ctx_type in (ast.Load, ast.Store, ast.Del):
            new_name = self._get_or_create_name(node.id)
            return ast.Name(id=new_name, ctx=ctx_type())
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Handle variable references and assignments."""
        return self._visit_name(node, type(node.ctx))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Normalize function definitions, including args and defaults."""
        new_name = self._get_or_create_name(node.name)
        node.name = new_name

        # Normalize argument names
        args = node.args
        for i, arg in enumerate(args.args):
            arg.arg = self._get_or_create_name(arg.arg)
        for i, arg in enumerate(args.posonlyargs):
            arg.arg = self._get_or_create_name(arg.arg)
        for i, arg in enumerate(args.kwonlyargs):
            arg.arg = self._get_or_create_name(arg.arg)
        if args.vararg:
            args.vararg.arg = self._get_or_create_name(args.vararg.arg)
        if args.kwarg:
            args.kwarg.arg = self._get_or_create_name(args.kwarg.arg)

        # Normalize default values
        node.args.defaults = [self.visit(d) for d in node.args.defaults]
        node.args.kw_defaults = [
            self.visit(d) if d is not None else None for d in node.args.kw_defaults
        ]

        # Visit body and decorators
        node.body = [self.visit(child) for child in node.body]
        node.decorator_list = [self.visit(child) for child in node.decorator_list]

        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        """Normalize class definitions."""
        new_name = self._get_or_create_name(node.name)
        node.name = new_name
        node.body = [self.visit(child) for child in node.body]
        node.decorator_list = [self.visit(child) for child in node.decorator_list]
        return node

    def visit_For(self, node: ast.For) -> ast.For:
        """Normalize For loop target variable and iterator."""
        if isinstance(node.target, ast.Name):
            node.target.id = self._get_or_create_name(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    elt.id = self._get_or_create_name(elt.id)
        node.iter = self.visit(node.iter)
        node.body = [self.visit(child) for child in node.body]
        node.orelse = [self.visit(child) for child in node.orelse]
        return node

    def visit_While(self, node: ast.While) -> ast.While:
        """Normalize While loop."""
        node.body = [self.visit(child) for child in node.body]
        node.orelse = [self.visit(child) for child in node.orelse]
        return node

    def visit_If(self, node: ast.If) -> ast.If:
        """Normalize If statement."""
        node.test = self.visit(node.test)
        node.body = [self.visit(child) for child in node.body]
        node.orelse = [self.visit(child) for child in node.orelse]
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.Assign:
        """Normalize assignment targets."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                target.id = self._get_or_create_name(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        elt.id = self._get_or_create_name(elt.id)
        node.value = self.visit(node.value)
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AnnAssign:
        """Normalize annotated assignment target."""
        if isinstance(node.target, ast.Name):
            node.target.id = self._get_or_create_name(node.target.id)
        if node.value:
            node.value = self.visit(node.value)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.ExceptHandler:
        """Normalize exception handler variable."""
        if node.name:
            node.name = self._get_or_create_name(node.name)
        node.body = [self.visit(child) for child in node.body]
        return node

    def visit_comprehension(self, node: ast.comprehension) -> ast.comprehension:
        """Normalize comprehension variables."""
        if isinstance(node.target, ast.Name):
            node.target.id = self._get_or_create_name(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    elt.id = self._get_or_create_name(elt.id)
        node.iter = self.visit(node.iter)
        node.ifs = [self.visit(if_clause) for if_clause in node.ifs]
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.ListComp:
        """Normalize list comprehensions."""
        node.elt = self.visit(node.elt)
        node.generators = [self.visit_comprehension(g) for g in node.generators]
        return node

    def visit_SetComp(self, node: ast.SetComp) -> ast.SetComp:
        """Normalize set comprehensions."""
        node.elt = self.visit(node.elt)
        node.generators = [self.visit_comprehension(g) for g in node.generators]
        return node

    def visit_DictComp(self, node: ast.DictComp) -> ast.DictComp:
        """Normalize dict comprehensions."""
        node.key = self.visit(node.key)
        node.value = self.visit(node.value)
        node.generators = [self.visit_comprehension(g) for g in node.generators]
        return node

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.NamedExpr:
        """Normalize named expression (walrus operator)."""
        if isinstance(node.target, ast.Name):
            node.target.id = self._get_or_create_name(node.target.id)
        node.value = self.visit(node.value)
        return node


def normalize_code(code: str) -> str:
    """Normalize all variable and function names in the code.

    Traverses the AST and renames all variables and functions
    sequentially in the order of appearance (e.g., var_1, var_2, var_3).

    Args:
        code: Python source code string to normalize.

    Returns:
        Normalized code string with sequential variable names,
        or the original code if parsing fails.
    """
    report = parse_code(code)

    if report.has_syntax_error or report.tree is None:
        return code

    try:
        normalizer = _VariableNormalizer()
        transformed_tree = normalizer.visit(report.tree)
        ast.fix_missing_locations(transformed_tree)
        return ast.unparse(transformed_tree)
    except Exception:
        return code


def get_normalized_signature(code: str) -> List[str]:
    """Extract the sequence of normalized variable/function names.

    Returns a list of normalized names in order of first appearance,
    e.g., ["var_1", "var_2", "var_3"].
    Returns the original code if parsing fails.
    """
    report = parse_code(code)

    if report.has_syntax_error or report.tree is None:
        return ["PARSING_ERROR"]

    try:
        normalizer = _VariableNormalizer()
        normalizer.visit(report.tree)
        return list(normalizer._name_map.values())
    except Exception:
        return ["NORMALIZATION_ERROR"]
