# Deploy to Render.com (free)

## Step 1: Push to GitHub

```bash
cd "C:\Users\Залман\Desktop\Новая папка"
git init
git add .
git commit -m "Football AI Predictor"
```

Create a new repo on github.com/new, then:
```bash
git remote add origin https://github.com/YOUR_USERNAME/football-ai.git
git push -u origin main
```

## Step 2: Deploy on Render

1. Go to render.com → Sign up (free)
2. Click "New +" → "Web Service"
3. Connect your GitHub repo
4. Settings:
   - Name: football-ai
   - Runtime: Python
   - Build: pip install -r requirements.txt
   - Start: uvicorn app:app --host 0.0.0.0 --port $PORT
5. Click "Create Web Service"
6. Wait 5-10 minutes (first build takes time)

## Step 3: Get your HTTPS URL

Render gives you: `https://football-ai.onrender.com`

This URL works with Telegram Mini App!

## Step 4: Connect Telegram Bot

Set environment variable in Render dashboard:
```
WEBAPP_URL=https://football-ai.onrender.com/mini-app
```

Or update Апи/telegram_token.txt and restart bot locally.

## Step 5: Open Mini App in Telegram

1. Open @LastOddsBot in Telegram
2. Send /start
3. The bot will show predictions
4. Open https://football-ai.onrender.com/mini-app in Telegram browser

## Notes

- Free tier spins down after 15 min of inactivity
- First request after sleep takes ~30 seconds
- Database resets on each deploy (collects fresh data)
- model.pkl is trained during build (adds ~5 min)
