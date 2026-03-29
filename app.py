import json
import os
import re
import socket
import ast
import builtins
import shutil
import subprocess
import tempfile
import textwrap
from typing import Any
from urllib.parse import urlparse

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")

SYSTEM_PROMPT = """You are a strict code review assistant.
Analyze the submitted code for the selected language.
Prioritize concrete bugs over style commentary.
Detect obvious syntax failures, undefined names, unreachable branches, broken conditions, unsafe execution, and suspicious API misuse.
Return valid JSON only with this exact schema:
{
  "summary": "short summary",
  "syntax_errors": [{"title": "string", "details": "string", "line": "string"}],
  "type_errors": [{"title": "string", "details": "string", "line": "string"}],
  "logical_issues": [{"title": "string", "details": "string", "line": "string"}],
  "security_issues": [{"title": "string", "details": "string", "line": "string"}],
  "suggested_fixes": [{"title": "string", "details": "string", "example": "string"}],
  "execution_trace": "string"
}
If no issues are found in a category, return an empty array for that category.
Keep findings concrete and concise.
"""

PYTHON_BUILTINS = set(dir(builtins))
RUFF_BIN = shutil.which("ruff")
MYPY_BIN = shutil.which("mypy")


class ReviewRequest(BaseModel):
    language: str = Field(..., pattern="^(Python|JavaScript|C\\+\\+)$")
    code: str


class ReviewResponse(BaseModel):
    summary: str
    syntax_errors: list[dict[str, str]]
    type_errors: list[dict[str, str]]
    logical_issues: list[dict[str, str]]
    security_issues: list[dict[str, str]]
    suggested_fixes: list[dict[str, str]]
    execution_trace: str = ""


app = FastAPI(title="Code Review")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def build_prompt(language: str, code: str) -> str:
    return (
        f"Language: {language}\n"
        "Review the following code and identify syntax errors, type errors, "
        "logical issues, security issues, and suggested fixes.\n\n"
        f"Code:\n```{language.lower()}\n{code}\n```"
    )


def normalize_review(payload: dict[str, Any]) -> ReviewResponse:
    return ReviewResponse(
        summary=str(payload.get("summary") or "Review completed."),
        syntax_errors=_normalize_items(payload.get("syntax_errors")),
        type_errors=_normalize_items(payload.get("type_errors")),
        logical_issues=_normalize_items(payload.get("logical_issues")),
        security_issues=_normalize_items(payload.get("security_issues")),
        suggested_fixes=_normalize_fixes(payload.get("suggested_fixes")),
        execution_trace=str(payload.get("execution_trace") or ""),
    )


def merge_model_review(base: ReviewResponse, model_review: ReviewResponse) -> ReviewResponse:
    merged = ReviewResponse(
        summary=base.summary,
        syntax_errors=list(base.syntax_errors),
        type_errors=list(base.type_errors),
        logical_issues=_merge_issue_lists(base.logical_issues, model_review.logical_issues),
        security_issues=_merge_issue_lists(base.security_issues, model_review.security_issues),
        suggested_fixes=_merge_fix_lists(base.suggested_fixes, model_review.suggested_fixes),
        execution_trace=base.execution_trace,
    )

    if model_review.summary and model_review.summary not in {"Review completed.", "No issues detected."}:
        if merged.summary == "No issues detected.":
            merged.summary = model_review.summary
        elif model_review.summary != merged.summary:
            merged.summary = f"{merged.summary} {model_review.summary}".strip()

    return merged


def _normalize_items(items: Any) -> list[dict[str, str]]:
    normalized = []
    for item in items or []:
        normalized.append(
            {
                "title": str(item.get("title") or "Issue"),
                "details": str(item.get("details") or "No details provided."),
                "line": str(item.get("line") or "N/A"),
            }
        )
    return normalized


def _normalize_fixes(items: Any) -> list[dict[str, str]]:
    normalized = []
    for item in items or []:
        normalized.append(
            {
                "title": str(item.get("title") or "Suggested fix"),
                "details": str(item.get("details") or "No details provided."),
                "example": str(item.get("example") or ""),
            }
        )
    return normalized


def _merge_issue_lists(
    base_items: list[dict[str, str]],
    extra_items: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged = [dict(item) for item in base_items]
    seen = {
        (
            str(item.get("line") or "N/A"),
            _normalize_issue_text(str(item.get("details") or "")),
        )
        for item in merged
    }
    for item in extra_items:
        key = (
            str(item.get("line") or "N/A"),
            _normalize_issue_text(str(item.get("details") or "")),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "title": str(item.get("title") or "Issue"),
                "details": str(item.get("details") or "No details provided."),
                "line": str(item.get("line") or "N/A"),
            }
        )
    return merged


def _merge_fix_lists(
    base_items: list[dict[str, str]],
    extra_items: list[dict[str, str]],
) -> list[dict[str, str]]:
    merged = [dict(item) for item in base_items]
    seen = {
        (
            str(item.get("title") or ""),
            str(item.get("details") or ""),
            str(item.get("example") or ""),
        )
        for item in merged
    }
    for item in extra_items:
        candidate = (
            str(item.get("title") or ""),
            str(item.get("details") or ""),
            str(item.get("example") or ""),
        )
        if candidate in seen:
            continue
        seen.add(candidate)
        merged.append(
            {
                "title": candidate[0] or "Suggested fix",
                "details": candidate[1] or "No details provided.",
                "example": candidate[2],
            }
        )
    return merged


def extract_json_block(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise HTTPException(status_code=502, detail="Model did not return valid JSON.")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=502, detail="Unable to parse model response.") from exc


def is_ollama_available() -> bool:
    parsed = urlparse(OLLAMA_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434

    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def normalize_submitted_code(language: str, code: str) -> str:
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip("\ufeff")
    if language == "Python":
        normalized = textwrap.dedent(normalized)
    return normalized.strip("\n")


def run_local_review(language: str, code: str, reason: str | None = None) -> ReviewResponse:
    code = normalize_submitted_code(language, code)
    syntax_errors: list[dict[str, str]] = []
    type_errors: list[dict[str, str]] = []
    logical_issues: list[dict[str, str]] = []
    security_issues: list[dict[str, str]] = []
    suggested_fixes: list[dict[str, str]] = []
    execution_trace = ""
    seen_items: set[tuple[str, str, str, str]] = set()
    seen_fixes: set[tuple[str, str, str]] = set()

    lines = code.splitlines()

    def add_item(target: list[dict[str, str]], title: str, details: str, line: str) -> None:
        key = (str(id(target)), title, details, line)
        if key in seen_items:
            return
        seen_items.add(key)
        target.append({"title": title, "details": details, "line": line})

    def add_fix(title: str, details: str, example: str = "") -> None:
        key = (title, details, example)
        if key in seen_fixes:
            return
        seen_fixes.add(key)
        suggested_fixes.append({"title": title, "details": details, "example": example})

    if language == "Python":
        try:
            tree = ast.parse(code, filename="<review>", mode="exec")
        except SyntaxError as exc:
            add_item(
                syntax_errors,
                "Python syntax error",
                exc.msg or "Python could not parse this code.",
                str(exc.lineno or "N/A"),
            )
            tree = None

        if tree is not None:
            findings = analyze_python_code(tree)
            syntax_errors.extend(findings["syntax_errors"])
            type_errors.extend(findings["type_errors"])
            logical_issues.extend(findings["logical_issues"])
            security_issues.extend(findings["security_issues"])
            suggested_fixes.extend(findings["suggested_fixes"])
            tool_findings = run_python_static_tools(code)
            for item in tool_findings["syntax_errors"]:
                add_item(syntax_errors, item["title"], item["details"], item["line"])
            for item in tool_findings["type_errors"]:
                add_item(type_errors, item["title"], item["details"], item["line"])
            for item in tool_findings["logical_issues"]:
                add_item(logical_issues, item["title"], item["details"], item["line"])
            for item in tool_findings["security_issues"]:
                add_item(security_issues, item["title"], item["details"], item["line"])
            runtime_items, runtime_trace = run_python_runtime_check(code)
            execution_trace = runtime_trace or execution_trace
            for item in runtime_items:
                add_item(type_errors, item["title"], item["details"], item["line"])

    if language == "JavaScript":
        syntax_errors.extend(run_external_syntax_check(language, code))
        runtime_items, runtime_trace = run_javascript_runtime_check(code)
        execution_trace = runtime_trace or execution_trace
        for item in runtime_items:
            add_item(type_errors, item["title"], item["details"], item["line"])
        for index, line in enumerate(lines, start=1):
            if re.search(r"\b(var)\b", line):
                add_item(
                    logical_issues,
                    "Prefer block-scoped declarations",
                    "var is function-scoped and can cause accidental reuse or hoisting issues.",
                    str(index),
                )
                add_fix(
                    "Use let or const",
                    "Use const by default and let only when reassignment is required.",
                    line.replace("var ", "const ", 1),
                )

            if "eval(" in line:
                add_item(
                    security_issues,
                    "Unsafe dynamic execution",
                    "eval() can execute arbitrary code and should not be used with external input.",
                    str(index),
                )

            if re.search(r"\b(if|while)\s*\([^)]*=[^=].*\)", line):
                add_item(
                    logical_issues,
                    "Assignment inside condition",
                    "This condition appears to assign a value instead of comparing one.",
                    str(index),
                )

            if "==" in line and "===" not in line:
                add_item(
                    logical_issues,
                    "Loose equality",
                    "== allows coercion and can hide simple bugs. Prefer === unless coercion is intentional.",
                    str(index),
                )

        open_braces = code.count("{")
        close_braces = code.count("}")
        if open_braces != close_braces:
            add_item(
                syntax_errors,
                "Possibly unbalanced braces",
                "The number of opening and closing braces does not match.",
                "N/A",
            )

    if language == "C++":
        cpp_findings, runtime_trace = run_cpp_checks(code)
        execution_trace = runtime_trace or execution_trace
        for item in cpp_findings["syntax_errors"]:
            add_item(syntax_errors, item["title"], item["details"], item["line"])
        for item in cpp_findings["type_errors"]:
            add_item(type_errors, item["title"], item["details"], item["line"])
        for index, line in enumerate(lines, start=1):
            if "gets(" in line or "strcpy(" in line:
                add_item(
                    security_issues,
                    "Unsafe standard library call",
                    "This function can overflow buffers and should be replaced with a safer alternative.",
                    str(index),
                )

            if re.search(r"\bif\s*\([^)]*=[^=].*\)", line):
                add_item(
                    logical_issues,
                    "Assignment inside condition",
                    "This condition appears to assign a value instead of comparing one.",
                    str(index),
                )

            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "//")) and re.search(r"\b(std::cout|return|int |float |double |char |bool |auto )", stripped):
                if not stripped.endswith((";", "{", "}")):
                    add_item(
                        syntax_errors,
                        "Possible missing semicolon",
                        "This line looks like a statement or declaration without a terminating semicolon.",
                        str(index),
                    )

    if language == "Python":
        syntax_errors = _dedupe_review_items(syntax_errors)
        type_errors = _dedupe_review_items(type_errors)
        logical_issues = _dedupe_review_items(logical_issues)
        security_issues = _dedupe_review_items(security_issues)

    total_findings = sum(
        len(bucket) for bucket in [syntax_errors, type_errors, logical_issues, security_issues]
    )
    summary = "Review completed."
    if total_findings == 0:
        summary = "No issues detected."

    return ReviewResponse(
        summary=summary,
        syntax_errors=syntax_errors,
        type_errors=type_errors,
        logical_issues=logical_issues,
        security_issues=security_issues,
        suggested_fixes=suggested_fixes,
        execution_trace=execution_trace,
    )


class PythonReviewVisitor(ast.NodeVisitor):
    def __init__(self, module_bindings: set[str]) -> None:
        self.syntax_errors: list[dict[str, str]] = []
        self.type_errors: list[dict[str, str]] = []
        self.logical_issues: list[dict[str, str]] = []
        self.security_issues: list[dict[str, str]] = []
        self.suggested_fixes: list[dict[str, str]] = []
        self.scope_stack: list[set[str]] = [set(PYTHON_BUILTINS) | module_bindings]
        self.constant_stack: list[dict[str, Any]] = [{}]
        self.seen_findings: set[tuple[str, str, str]] = set()
        self.function_depth = 0

    def add_issue(
        self,
        target: list[dict[str, str]],
        title: str,
        details: str,
        line: int | str | None,
    ) -> None:
        line_value = str(line or "N/A")
        key = (title, details, line_value)
        if key in self.seen_findings:
            return
        self.seen_findings.add(key)
        target.append({"title": title, "details": details, "line": line_value})

    def add_fix(self, title: str, details: str, example: str = "") -> None:
        key = (title, details, example)
        if key in self.seen_findings:
            return
        self.seen_findings.add(key)
        self.suggested_fixes.append({"title": title, "details": details, "example": example})

    def current_scope(self) -> set[str]:
        return self.scope_stack[-1]

    def current_constants(self) -> dict[str, Any]:
        return self.constant_stack[-1]

    def define_name(self, name: str) -> None:
        self.current_scope().add(name)

    def is_defined(self, name: str) -> bool:
        return any(name in scope for scope in reversed(self.scope_stack))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.define_name(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name == "*":
                continue
            self.define_name(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        resolved = self._resolve_constant(node.value)
        self._detect_constant_assignment_issues(node.value, node.lineno, resolved)
        self._detect_self_reference_assignment(node.value, node.targets, node.lineno)
        for target in node.targets:
            self._register_target(target)
            self._track_constant_target(target, resolved)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        resolved = self._resolve_constant(node.value) if node.value is not None else None
        if node.value is not None:
            self._detect_constant_assignment_issues(node.value, node.lineno, resolved)
            self._detect_self_reference_assignment(node.value, [node.target], node.lineno)
        self._register_target(node.target)
        self._track_constant_target(node.target, resolved)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._register_target(node.target)
        self._track_constant_target(node.target, None)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._register_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._register_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._register_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.define_name(node.name)
        self.scope_stack.append(set(self.current_scope()))
        self.constant_stack.append(dict(self.current_constants()))
        self.generic_visit(node)
        self.constant_stack.pop()
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and not self.is_defined(node.id):
            self.add_issue(
                self.type_errors,
                "Possible undefined name",
                f"`{node.id}` is referenced before any visible definition in this scope.",
                node.lineno,
            )

    def visit_Subscript(self, node: ast.Subscript) -> None:
        sequence_value = self._resolve_constant(node.value)
        index_value = self._resolve_constant(node.slice)
        if isinstance(sequence_value, dict) and index_value is not None:
            if index_value not in sequence_value:
                self.add_issue(
                    self.logical_issues,
                    "Missing dictionary key",
                    f"This dictionary access uses the constant key `{index_value}`, but that key is not present.",
                    node.lineno,
                )
                self.add_fix(
                    "Guard dictionary access",
                    "Check that the key exists first or use dict.get() with a default value.",
                    "print(data.get(\"age\"))",
                )
        if isinstance(sequence_value, (list, tuple, str)) and isinstance(index_value, int):
            if not (-len(sequence_value) <= index_value < len(sequence_value)):
                self.add_issue(
                    self.logical_issues,
                    "Index out of range",
                    "A constant sequence is accessed with an index that exceeds its bounds.",
                    node.lineno,
                )
                self.add_fix(
                    "Guard index access",
                    "Check the sequence length before indexing or use a valid offset.",
                    "if len(arr) > 5:\n    print(arr[5])",
                )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        right = self._resolve_constant(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            self.add_issue(
                self.logical_issues,
                "Division by zero",
                "This expression always divides or mods by zero.",
                node.lineno,
            )
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        left = self._resolve_constant(node.left)
        comparators = [self._resolve_constant(item) for item in node.comparators]
        operands = [left, *comparators]
        if len(operands) == 2 and all(value is not None for value in operands):
            try:
                result = self._evaluate_compare(operands[0], node.ops[0], operands[1])
            except Exception:
                result = None
            if result is True:
                self.add_issue(
                    self.logical_issues,
                    "Always-true comparison",
                    "This comparison resolves to True with the constant values in the code.",
                    node.lineno,
                )
            elif result is False:
                self.add_issue(
                    self.logical_issues,
                    "Always-false comparison",
                    "This comparison resolves to False with the constant values in the code.",
                    node.lineno,
                )
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        test_value = self._resolve_constant(node.test)
        if isinstance(test_value, bool):
            title = "Dead branch" if test_value else "Unreachable branch"
            details = (
                "This condition is always True, so the else branch will never run."
                if test_value
                else "This condition is always False, so the body will never run."
            )
            self.add_issue(self.logical_issues, title, details, node.lineno)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call_name = self._call_name(node.func)
        if call_name in {"eval", "exec"}:
            self.add_issue(
                self.security_issues,
                "Unsafe dynamic execution",
                f"{call_name}() can execute untrusted input and should be avoided.",
                node.lineno,
            )

        if call_name == "pickle.loads":
            self.add_issue(
                self.security_issues,
                "Unsafe deserialization",
                "Unpickling untrusted data can execute arbitrary code.",
                node.lineno,
            )

        if call_name == "subprocess.run":
            for keyword in node.keywords:
                if keyword.arg == "shell" and self._resolve_constant(keyword.value) is True:
                    self.add_issue(
                        self.security_issues,
                        "Shell execution enabled",
                        "subprocess.run(..., shell=True) is risky with dynamic input.",
                        node.lineno,
                    )

        if call_name in {"open", "len", "range", "print", "str", "int", "float"}:
            expected = {
                "open": (1, 8),
                "len": (1, 1),
                "range": (1, 3),
                "print": (0, None),
                "str": (0, 1),
                "int": (0, 2),
                "float": (0, 1),
            }[call_name]
            self._check_argument_count(call_name, node, *expected)

        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.define_name(node.name)
        function_scope = set(self.current_scope())
        local_bindings = self._collect_function_bindings(node)
        function_scope.update(local_bindings)
        function_scope.update(self._collect_argument_names(node.args))
        self.scope_stack.append(function_scope)
        self.constant_stack.append({})
        self.function_depth += 1

        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self.add_issue(
                    self.logical_issues,
                    "Mutable default argument",
                    "A mutable default value is shared across calls.",
                    getattr(default, "lineno", node.lineno),
                )
                self.add_fix(
                    "Replace mutable defaults",
                    "Use None and create the mutable object inside the function.",
                    "def add_item(item, bucket=None):\n    if bucket is None:\n        bucket = []\n    bucket.append(item)\n    return bucket",
                )

        self.generic_visit(node)
        self.function_depth -= 1
        self.constant_stack.pop()
        self.scope_stack.pop()

    def _call_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = self._call_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return None

    def _collect_function_bindings(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
        bindings: set[str] = set()
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bindings.add(child.name)
        return bindings

    def _collect_argument_names(self, args: ast.arguments) -> set[str]:
        names = {arg.arg for arg in args.posonlyargs + args.args + args.kwonlyargs}
        if args.vararg:
            names.add(args.vararg.arg)
        if args.kwarg:
            names.add(args.kwarg.arg)
        return names

    def _register_target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.define_name(node.id)
            return
        for child in ast.iter_child_nodes(node):
            self._register_target(child)

    def _track_constant_target(self, node: ast.AST, value: Any) -> None:
        if isinstance(node, ast.Name):
            if value is None:
                self.current_constants().pop(node.id, None)
            else:
                self.current_constants()[node.id] = value
            return
        for child in ast.iter_child_nodes(node):
            self._track_constant_target(child, None)

    def _resolve_constant(self, node: ast.AST | None) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Name):
            for constants in reversed(self.constant_stack):
                if node.id in constants:
                    return constants[node.id]
            return None
        if isinstance(node, ast.UnaryOp):
            operand = self._resolve_constant(node.operand)
            if operand is None:
                return None
            if isinstance(node.op, ast.Not):
                return not operand
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
        if isinstance(node, ast.BoolOp):
            values = [self._resolve_constant(value) for value in node.values]
            if any(value is None for value in values):
                return None
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            left = self._resolve_constant(node.left)
            right = self._resolve_constant(node.comparators[0])
            if left is not None and right is not None:
                try:
                    return self._evaluate_compare(left, node.ops[0], right)
                except Exception:
                    return None
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    def _detect_constant_assignment_issues(self, value: ast.AST, line: int, resolved: Any) -> None:
        if isinstance(resolved, dict) and len(resolved) == 1:
            if isinstance(value, ast.Dict) and len(value.keys) > 1:
                keys = [self._resolve_constant(key) for key in value.keys]
                unique_keys = {key for key in keys if key is not None}
                if len(unique_keys) < len([key for key in keys if key is not None]):
                    self.add_issue(
                        self.logical_issues,
                        "Duplicate dictionary key",
                        "A later constant key overwrites an earlier entry in this dictionary literal.",
                        line,
                    )

    def _detect_self_reference_assignment(
        self,
        value: ast.AST,
        targets: list[ast.AST],
        line: int,
    ) -> None:
        assigned_names = {target.id for target in targets if isinstance(target, ast.Name)}
        if not assigned_names:
            return
        for child in ast.walk(value):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in assigned_names:
                self.add_issue(
                    self.type_errors,
                    "Name used before assignment",
                    f"`{child.id}` is read while being assigned on the same statement.",
                    line,
                )

    def _evaluate_compare(self, left: Any, op: ast.cmpop, right: Any) -> bool | None:
        operations = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
            ast.Is: lambda a, b: a is b,
            ast.IsNot: lambda a, b: a is not b,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }
        for op_type, fn in operations.items():
            if isinstance(op, op_type):
                return fn(left, right)
        return None

    def _check_argument_count(
        self,
        call_name: str,
        node: ast.Call,
        minimum: int,
        maximum: int | None,
    ) -> None:
        if any(keyword.arg is None for keyword in node.keywords):
            return
        count = len(node.args) + len(node.keywords)
        if count < minimum or (maximum is not None and count > maximum):
            expected = f"{minimum}" if minimum == maximum else f"{minimum}-{maximum}" if maximum is not None else f"{minimum}+"
            self.add_issue(
                self.type_errors,
                "Wrong argument count",
                f"{call_name}() is called with {count} argument(s), but the common valid range is {expected}.",
                node.lineno,
            )


def run_external_syntax_check(language: str, code: str) -> list[dict[str, str]]:
    if language == "JavaScript":
        checker = shutil.which("node")
        if checker:
            return _run_temp_check(
                [checker, "--check"],
                code,
                ".js",
                language,
            )
    if language == "C++":
        checker = shutil.which("g++") or shutil.which("clang++")
        if checker:
            return _run_temp_check(
                [checker, "-fsyntax-only"],
                code,
                ".cpp",
                language,
            )
    return []


def run_python_static_tools(code: str) -> dict[str, list[dict[str, str]]]:
    findings = {"syntax_errors": [], "type_errors": [], "logical_issues": [], "security_issues": []}
    if not (RUFF_BIN or MYPY_BIN):
        return findings

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(code)
        temp_path = handle.name

    try:
        if RUFF_BIN:
            findings = _merge_findings(findings, _run_ruff_check(temp_path))
        if MYPY_BIN:
            findings = _merge_findings(findings, _run_mypy_check(temp_path))
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass
    return findings


def run_python_runtime_check(code: str) -> tuple[list[dict[str, str]], str]:
    checker = shutil.which("python3")
    if not checker:
        return [], ""
    result = _run_temp_command([checker, "-I", "-X", "faulthandler"], code, ".py")
    if result is None or result.returncode == 0:
        return [], ""
    output = (result.stderr or result.stdout).strip()
    line_match = re.findall(r'File ".*?", line (\d+)', output)
    details = _extract_error_line(output) or "Python raised a runtime error."
    if result.returncode == 124:
        details = "Python execution timed out during review."
    error_name = _extract_exception_name(details) or "Python runtime error"
    return (
        [
            {
                "title": error_name,
                "details": details,
                "line": line_match[-1] if line_match else "N/A",
            }
        ],
        output,
    )


def _run_ruff_check(path: str) -> dict[str, list[dict[str, str]]]:
    findings = {"syntax_errors": [], "type_errors": [], "logical_issues": [], "security_issues": []}
    try:
        result = subprocess.run(
            [
                RUFF_BIN,
                "check",
                "--select",
                "F,E9,B,PLE,PLW,S",
                "--output-format",
                "json",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=6,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return findings

    output = (result.stdout or "").strip()
    if not output:
        return findings

    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return findings

    for item in payload:
        code = str(item.get("code") or "Ruff")
        message = str(item.get("message") or "Ruff detected an issue.")
        location = item.get("location") or {}
        line = str(location.get("row") or "N/A")
        title = f"{code} ({_ruff_rule_label(code)})"
        bucket = _ruff_bucket(code)
        findings[bucket].append({"title": title, "details": message, "line": line})
    return findings


def _run_mypy_check(path: str) -> dict[str, list[dict[str, str]]]:
    findings = {"syntax_errors": [], "type_errors": [], "logical_issues": [], "security_issues": []}
    try:
        result = subprocess.run(
            [
                MYPY_BIN,
                "--check-untyped-defs",
                "--warn-unreachable",
                "--strict-equality",
                "--extra-checks",
                "--show-error-codes",
                "--show-column-numbers",
                "--hide-error-context",
                "--no-color-output",
                "--no-error-summary",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return findings

    output = (result.stdout or result.stderr or "").strip()
    if not output:
        return findings

    for line in output.splitlines():
        match = re.match(r"^(.*?):(\d+):(?:(\d+):)?\s*(error|note):\s*(.*)$", line.strip())
        if not match:
            continue
        _, line_number, _, severity, message = match.groups()
        bucket = "syntax_errors" if "syntax" in message.lower() else "type_errors"
        title = "Mypy error" if severity == "error" else "Mypy note"
        findings[bucket].append(
            {
                "title": title,
                "details": message,
                "line": line_number,
            }
        )
    return findings


def _merge_findings(
    left: dict[str, list[dict[str, str]]],
    right: dict[str, list[dict[str, str]]],
) -> dict[str, list[dict[str, str]]]:
    for key in left:
        left[key].extend(right.get(key, []))
    return left


def _ruff_bucket(code: str) -> str:
    if code.startswith("E9"):
        return "syntax_errors"
    if code.startswith("S"):
        return "security_issues"
    if code.startswith("B"):
        return "logical_issues"
    return "type_errors"


def _ruff_rule_label(code: str) -> str:
    if code.startswith("E9"):
        return "syntax"
    if code.startswith("F"):
        return "pyflakes"
    if code.startswith("B"):
        return "bugbear"
    if code.startswith("S"):
        return "bandit"
    if code.startswith("PLE"):
        return "pylint error"
    if code.startswith("PLW"):
        return "pylint warning"
    return "ruff"


def run_javascript_runtime_check(code: str) -> tuple[list[dict[str, str]], str]:
    checker = shutil.which("node")
    if not checker:
        return [], ""
    result = _run_temp_command([checker], code, ".js")
    if result is None or result.returncode == 0:
        return [], ""
    output = (result.stderr or result.stdout).strip()
    line_match = re.search(r":(\d+)\b", output)
    details = _extract_error_line(output) or "JavaScript raised a runtime error."
    if result.returncode == 124:
        details = "JavaScript execution timed out during review."
    error_name = _extract_exception_name(details) or "JavaScript runtime error"
    return (
        [
            {
                "title": error_name,
                "details": details,
                "line": line_match.group(1) if line_match else "N/A",
            }
        ],
        output,
    )


def run_cpp_checks(code: str) -> tuple[dict[str, list[dict[str, str]]], str]:
    compiler = shutil.which("g++") or shutil.which("clang++")
    findings = {"syntax_errors": [], "type_errors": []}
    if not compiler:
        return findings, ""

    with tempfile.TemporaryDirectory() as temp_dir:
        source_path = os.path.join(temp_dir, "review.cpp")
        output_path = os.path.join(temp_dir, "review.out")
        try:
            with open(source_path, "w", encoding="utf-8") as handle:
                handle.write(code)
            compile_result = subprocess.run(
                [compiler, "-std=c++17", "-Wall", "-Wextra", source_path, "-o", output_path],
                capture_output=True,
                text=True,
                timeout=6,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return findings, ""

        if compile_result.returncode != 0:
            output = (compile_result.stderr or compile_result.stdout).strip()
            line_match = re.search(r":(\d+)(?::\d+)?:", output)
            findings["syntax_errors"].append(
                {
                    "title": "C++ compiler error",
                    "details": _first_nonempty_line(output) or "C++ compilation failed.",
                    "line": line_match.group(1) if line_match else "N/A",
                }
            )
            return findings, output

        try:
            runtime_result = subprocess.run(
                [output_path],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
                cwd=temp_dir,
            )
        except subprocess.TimeoutExpired:
            findings["type_errors"].append(
                {
                    "title": "C++ runtime timeout",
                    "details": "Compiled code did not finish within the review timeout.",
                    "line": "N/A",
                }
            )
            return findings, "Execution timed out."
        except OSError:
            return findings, ""

        if runtime_result.returncode != 0:
            output = (runtime_result.stderr or runtime_result.stdout).strip()
            findings["type_errors"].append(
                {
                    "title": _extract_exception_name(_extract_error_line(output) or "") or "C++ runtime error",
                    "details": _extract_error_line(output) or "Compiled code exited with an error.",
                    "line": "N/A",
                }
            )
            return findings, output
    return findings, ""


def _run_temp_command(command: list[str], code: str, suffix: str) -> subprocess.CompletedProcess[str] | None:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8") as handle:
            handle.write(code)
            temp_path = handle.name
        try:
            return subprocess.run(
                [*command, temp_path],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess([*command, temp_path], 124, "", "")
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_error_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if "Error:" in stripped:
            return stripped
    return _last_nonempty_line(text)


def _extract_exception_name(text: str) -> str:
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*Error|[A-Za-z_][A-Za-z0-9_]*Exception|ZeroDivisionError|KeyError|NameError|TypeError|ValueError|RuntimeError|TimeoutError|SyntaxError)\b", text.strip())
    return match.group(1) if match else ""


def _dedupe_review_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: dict[tuple[str, str], dict[str, str]] = {}

    for item in items:
        normalized = _normalize_issue_text(item.get("details", ""))
        line = str(item.get("line") or "N/A")
        key = (line, normalized)
        current = {
            "title": str(item.get("title") or "Issue"),
            "details": str(item.get("details") or "No details provided."),
            "line": line,
        }

        if key not in seen:
            seen[key] = current
            deduped.append(current)
            continue

        if _issue_priority(current) > _issue_priority(seen[key]):
            existing = seen[key]
            existing["title"] = current["title"]
            existing["details"] = current["details"]

    return deduped


def _normalize_issue_text(text: str) -> str:
    normalized = text.lower().strip()
    undefined_name = _extract_named_symbol(normalized)
    if undefined_name:
        return f"undefined-name:{undefined_name}"

    runtime_missing_key_match = re.search(r"keyerror:\s*[\"'`]?([^\"'`\s]+)[\"'`]?", normalized)
    if runtime_missing_key_match:
        return f"missing-key:{runtime_missing_key_match.group(1)}"

    static_missing_key_match = re.search(r"constant key\s+[\"'`]?([^\"'`\s]+)[\"'`]?\s*,?\s+but that key is not present", normalized)
    if static_missing_key_match:
        return f"missing-key:{static_missing_key_match.group(1)}"

    normalized = re.sub(r"`([^`]+)`", r"\1", normalized)
    normalized = re.sub(r'"([^"]+)"', r"\1", normalized)
    normalized = re.sub(r"'([^']+)'", r"\1", normalized)
    normalized = normalized.replace(" is not defined", "")
    normalized = normalized.replace("undefined name ", "")
    normalized = normalized.replace("name ", "")
    normalized = normalized.replace("cannot read properties of ", "")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _issue_priority(item: dict[str, str]) -> int:
    title = str(item.get("title") or "")
    if title in {"SyntaxError", "TypeError", "NameError", "KeyError", "ValueError", "RuntimeError"}:
        return 4
    if title.startswith("Mypy"):
        return 3
    if title.startswith("F") or title.startswith("E9") or title.startswith("B") or title.startswith("PLE") or title.startswith("PLW") or title.startswith("S"):
        return 2
    return 1


def _extract_named_symbol(text: str) -> str:
    patterns = [
        r"undefined name\s+[\"'`]?([a-zA-Z_][a-zA-Z0-9_]*)[\"'`]?(?:\s|$)",
        r"name\s+[\"'`]?([a-zA-Z_][a-zA-Z0-9_]*)[\"'`]?\s+is not defined",
        r"[\"'`]?([a-zA-Z_][a-zA-Z0-9_]*)[\"'`]?\s+is referenced before any visible definition in this scope",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return ""


def _run_temp_check(
    command: list[str],
    code: str,
    suffix: str,
    language: str,
) -> list[dict[str, str]]:
    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as handle:
            handle.write(code)
            temp_path = handle.name
        result = subprocess.run(
            [*command, temp_path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass

    if result.returncode == 0:
        return []

    message = (result.stderr or result.stdout).strip()
    if not message:
        message = f"{language} syntax check failed."
    line_match = re.search(r":(\d+)(?::\d+)?:", message)
    return [
        {
            "title": f"{language} syntax error",
            "details": message.splitlines()[0],
            "line": line_match.group(1) if line_match else "N/A",
        }
    ]


def analyze_python_code(tree: ast.AST) -> dict[str, list[dict[str, str]]]:
    visitor = PythonReviewVisitor(collect_module_bindings(tree))
    visitor.visit(tree)
    return {
        "syntax_errors": visitor.syntax_errors,
        "type_errors": visitor.type_errors,
        "logical_issues": visitor.logical_issues,
        "security_issues": visitor.security_issues,
        "suggested_fixes": visitor.suggested_fixes,
    }


def collect_module_bindings(tree: ast.AST) -> set[str]:
    bindings: set[str] = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    bindings.add(alias.asname or alias.name)
    return bindings


async def query_ollama(language: str, code: str) -> ReviewResponse:
    code = normalize_submitted_code(language, code)
    local_review = run_local_review(language, code)
    if not is_ollama_available():
        return local_review

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_prompt(language, code),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
    except (httpx.ConnectError, httpx.TimeoutException):
        return local_review
    except httpx.HTTPError:
        return local_review

    data = response.json()
    model_text = data.get("response", "")
    try:
        parsed = extract_json_block(model_text)
        return merge_model_review(local_review, normalize_review(parsed))
    except HTTPException:
        return local_review


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/review", response_model=ReviewResponse)
async def review_code(review_request: ReviewRequest) -> ReviewResponse:
    if not review_request.code.strip():
        raise HTTPException(status_code=400, detail="Please paste some code before running a review.")
    return await query_ollama(review_request.language, review_request.code)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "3033"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
