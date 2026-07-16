# DEPLOY.md — Неактуально

> **Этот файл больше не используется.** Публичный деплой не планируется.
> Для локального запуска см. `LOCAL_SETUP.md`.

---

Старая инструкция по деплою на Render.com сохранена ниже для справки.

<details>
<summary>Старая инструкция (Render.com)</summary>

## Deploy to Render.com (free)

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Football AI Predictor"
```

### Step 2: Deploy on Render

1. Go to render.com → Sign up (free)
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - Name: football-ai
   - Runtime: Python
   - Build: pip install -r requirements.txt
   - Start: uvicorn app:app --host 0.0.0.0 --port $PORT
5. Click "Create Web Service"

### Notes

- Free tier spins down after 15 min of inactivity
- Database resets on each deploy
- model.pkl is trained during build

</details>
