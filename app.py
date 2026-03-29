import json
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "tinyllama"

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


app = FastAPI(title="CodeBot Review App")
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


async def query_ollama(language: str, code: str) -> ReviewResponse:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": build_prompt(language, code),
        "system": SYSTEM_PROMPT,
        "stream": False,
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail="Could not connect to Ollama. Make sure Ollama is running and tinyllama is available.",
        ) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Ollama request failed.") from exc

    data = response.json()
    model_text = data.get("response", "")
    parsed = extract_json_block(model_text)
    return normalize_review(parsed)


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
