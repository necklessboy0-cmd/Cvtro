# CVtro on Vercel (Python / Flask)

```
cvtro-vercel/
├─ app.py              Flask backend (/api/gemini), keeps your key private
├─ requirements.txt    flask
├─ vercel.json
└─ public/index.html   UI
```

## Deploy
1. Push this folder to GitHub (or run `vercel` inside it).
2. Vercel → Settings → Environment Variables → add `GEMINI_API_KEY`
   (optional: `GEMINI_MODEL`, `GEMINI_EMBED_MODEL`).
3. Redeploy.

## Local test
```
pip install -r requirements.txt
GEMINI_API_KEY=your_key python app.py     # http://localhost:3000
```
