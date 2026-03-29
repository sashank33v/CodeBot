import json
import os
import re
import socket
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
            compile(code, "<review>", "exec")
        except SyntaxError as exc:
            add_item(
                syntax_errors,
                "Python syntax error",
                exc.msg or "Python could not parse this code.",
                str(exc.lineno or "N/A"),
            )

        for index, line in enumerate(lines, start=1):
            if re.search(r"def\s+\w+\s*\([^)]*=\[\]\)", line):
                add_item(
                    logical_issues,
                    "Mutable default argument",
                    "Using [] as a default value shares the same list across calls.",
                    str(index),
                )
                add_fix(
                    "Use None as the default",
                    "Create the list inside the function so each call gets a fresh value.",
                    "def add_item(item, my_list=None):\n    if my_list is None:\n        my_list = []\n    my_list.append(item)\n    return my_list",
                )

            if "eval(" in line:
                add_item(
                    security_issues,
                    "Unsafe dynamic execution",
                    "eval() can execute untrusted input and should be avoided.",
                    str(index),
                )

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

    if not any([syntax_errors, type_errors, logical_issues, security_issues]):
        logical_issues.append(
            {
                "title": "No obvious static issues detected",
                "details": "The fallback review did not find a strong issue signal. For deeper analysis, connect Ollama with a stronger coding model.",
                "line": "N/A",
            }
        )

    summary = "Review completed with the local fallback analyzer."
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
