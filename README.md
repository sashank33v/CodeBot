# Code Review

A FastAPI web app for code review using Ollama when available, with a built-in local fallback analyzer.

## Features

- Paste code and choose Python, JavaScript, or C++
- Review for syntax errors, type errors, logical issues, and security issues
- Receive suggested fixes in a cleaner browser UI
- Falls back to a local analyzer when Ollama is unavailable or returns unusable output
- Uses Python AST checks, `ruff`, `mypy`, and runtime traceback capture for stronger Python diagnostics
- Shows an execution trace panel when runtime failures occur

## Run

1. Ensure Ollama is installed and the model is available:

   ```bash
   ollama pull tinyllama
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the app:

   ```bash
   python3 app.py
   ```

4. Open `http://localhost:3033`

## Deploy

The repo includes [render.yaml](/home/sashank/codebot/render.yaml) for Render.

1. Push the repository to GitHub.
2. Create a new Render Blueprint from the repo.
3. Render will install dependencies from `requirements.txt` and start `uvicorn app:app --host 0.0.0.0 --port $PORT`.
