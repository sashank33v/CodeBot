# Code Review

A FastAPI web app for code review using Ollama when available, with a built-in local fallback analyzer.

## Features

- Paste code and choose Python, JavaScript, or C++
- Review for syntax errors, type errors, logical issues, and security issues
- Receive suggested fixes in a cleaner browser UI
- Falls back to a local analyzer when Ollama is unavailable or returns unusable output

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
