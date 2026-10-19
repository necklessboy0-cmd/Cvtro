"""CVtro backend for Vercel (Flask). Keeps GEMINI_API_KEY server-side.

Routes
  POST /api/gemini   {op:"generate", prompt}  -> {text}
                     {op:"embed", texts, task} -> {vectors}
  GET  /             serves public/index.html (local dev; Vercel serves public/ itself)

Env vars: GEMINI_API_KEY (required), GEMINI_MODEL, GEMINI_EMBED_MODEL
Local run: pip install -r requirements.txt && GEMINI_API_KEY=... python app.py
"""
import json
import os
import urllib.error
import urllib.request

from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 1_000_000  # 1 MB request cap

GEN_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
PUBLIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")


def call_gemini(path: str, payload: dict, key: str) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=55) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode("utf-8"))
        except Exception:
            body = {}
        return e.code, {"error": (body.get("error") or {}).get("message", "Gemini error")}


@app.post("/api/gemini")
def gemini():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return jsonify(error="GEMINI_API_KEY is not set on the server."), 500
    data = request.get_json(silent=True) or {}
    op = data.get("op")

    if op == "generate":
        prompt = data.get("prompt")
        if not isinstance(prompt, str) or not prompt or len(prompt) > 80_000:
            return jsonify(error="Bad prompt."), 400
        status, j = call_gemini(
            f"{GEN_MODEL}:generateContent",
            {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseMimeType": "application/json", "temperature": 0.2},
            },
            key,
        )
        if status != 200:
            return jsonify(error=j.get("error", "Gemini error")), status
        parts = (((j.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
        return jsonify(text="".join(p.get("text", "") for p in parts))

    if op == "embed":
        texts = data.get("texts")
        if (
            not isinstance(texts, list)
            or not texts
            or len(texts) > 64
            or any(not isinstance(t, str) or len(t) > 8000 for t in texts)
        ):
            return jsonify(error="Bad texts."), 400
        task = "RETRIEVAL_QUERY" if data.get("task") == "RETRIEVAL_QUERY" else "RETRIEVAL_DOCUMENT"
        status, j = call_gemini(
            f"{EMBED_MODEL}:batchEmbedContents",
            {
                "requests": [
                    {
                        "model": f"models/{EMBED_MODEL}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                        "outputDimensionality": 768,
                    }
                    for t in texts
                ]
            },
            key,
        )
        if status != 200:
            return jsonify(error=j.get("error", "Embedding error")), status
        return jsonify(vectors=[e["values"] for e in j.get("embeddings", [])])

    return jsonify(error="Unknown op."), 400


@app.get("/")
def index():
    return send_from_directory(PUBLIC, "index.html")


if __name__ == "__main__":
    app.run(port=int(os.getenv("PORT", "3000")), debug=True)
