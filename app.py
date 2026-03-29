import json
import os
import re
import socket
import ast
import builtins
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
Return valid JSON only with this exact schema:
{
  "summary": "short summary",
  "syntax_errors": [{"title": "string", "details": "string", "line": "string"}],
  "type_errors": [{"title": "string", "details": "string", "line": "string"}],
  "logical_issues": [{"title": "string", "details": "string", "line": "string"}],
  "security_issues": [{"title": "string", "details": "string", "line": "string"}],
  "suggested_fixes": [{"title": "string", "details": "string", "example": "string"}]
}
If no issues are found in a category, return an empty array for that category.
Keep findings concrete and concise.
"""

PYTHON_BUILTINS = set(dir(builtins))


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
    )


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


def run_local_review(language: str, code: str, reason: str | None = None) -> ReviewResponse:
    syntax_errors: list[dict[str, str]] = []
    type_errors: list[dict[str, str]] = []
    logical_issues: list[dict[str, str]] = []
    security_issues: list[dict[str, str]] = []
    suggested_fixes: list[dict[str, str]] = []

    lines = code.splitlines()

    def add_item(target: list[dict[str, str]], title: str, details: str, line: str) -> None:
        target.append({"title": title, "details": details, "line": line})

    def add_fix(title: str, details: str, example: str = "") -> None:
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

    if language == "JavaScript":
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
        for index, line in enumerate(lines, start=1):
            if "gets(" in line or "strcpy(" in line:
                add_item(
                    security_issues,
                    "Unsafe standard library call",
                    "This function can overflow buffers and should be replaced with a safer alternative.",
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

    total_findings = sum(
        len(bucket) for bucket in [syntax_errors, type_errors, logical_issues, security_issues]
    )
    summary = "Review completed with the local fallback analyzer."
    if total_findings == 0:
        summary = "Review completed. The local analyzer did not detect a concrete issue in this snippet."
    if reason:
        summary = f"{summary} {reason}"

    return ReviewResponse(
        summary=summary,
        syntax_errors=syntax_errors,
        type_errors=type_errors,
        logical_issues=logical_issues,
        security_issues=security_issues,
        suggested_fixes=suggested_fixes,
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
        for target in node.targets:
            self._register_target(target)
            self._track_constant_target(target, resolved)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        resolved = self._resolve_constant(node.value) if node.value is not None else None
        if node.value is not None:
            self._detect_constant_assignment_issues(node.value, node.lineno, resolved)
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

        self.generic_visit(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.define_name(node.name)
        function_scope = set(self.current_scope())
        local_bindings = self._collect_function_bindings(node)
        function_scope.update(local_bindings)
        function_scope.update(self._collect_argument_names(node.args))
        self.scope_stack.append(function_scope)
        self.constant_stack.append({})

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
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                bindings.add(child.id)
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
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)):
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    bindings.add(child.id)
    return bindings


async def query_ollama(language: str, code: str) -> ReviewResponse:
    if not is_ollama_available():
        return run_local_review(
            language,
            code,
            f"Ollama was unavailable at {OLLAMA_URL}.",
        )

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
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        return run_local_review(
            language,
            code,
            f"Ollama was unavailable at {OLLAMA_URL}.",
        )
    except httpx.HTTPError as exc:
        return run_local_review(
            language,
            code,
            "Ollama returned an error response.",
        )

    data = response.json()
    model_text = data.get("response", "")
    try:
        parsed = extract_json_block(model_text)
        return normalize_review(parsed)
    except HTTPException:
        return run_local_review(
            language,
            code,
            "The model response could not be parsed cleanly, so a local fallback review was used.",
        )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/review", response_model=ReviewResponse)
async def review_code(review_request: ReviewRequest) -> ReviewResponse:
    if not review_request.code.strip():
        raise HTTPException(status_code=400, detail="Please paste some code before running a review.")
    return await query_ollama(review_request.language, review_request.code)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=3033, reload=False)
