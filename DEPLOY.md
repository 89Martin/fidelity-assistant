# Deploying to the web (Streamlit Community Cloud)

This puts your app at a public URL like `https://your-app.streamlit.app` — no
Python needed by visitors, and your API key stays private.

## One-time setup

You need (both free):
- A **GitHub** account — https://github.com/join
- A **Streamlit Community Cloud** account — https://share.streamlit.io
  (sign in with the same GitHub account)

## Step 1 — Put the code on GitHub

From this folder (`C:\Users\asimm\OneDrive\Documents\Stock`):

```powershell
git init
git add .
git commit -m "Fidelity Wealth & Debt Assistant"
```

Create an empty repo on GitHub (e.g. `fidelity-assistant`), then:

```powershell
git branch -M main
git remote add origin https://github.com/<your-username>/fidelity-assistant.git
git push -u origin main
```

> The `.gitignore` already excludes `.env` and `secrets.toml`, so your key is
> NOT uploaded. Good.

## Step 2 — Deploy

1. Go to https://share.streamlit.io and click **"Create app"**.
2. Pick your GitHub repo, branch `main`, main file `app.py`.
3. Click **Advanced settings → Secrets** and paste exactly:

   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-REAL-key-here"
   ```

4. Click **Deploy**. First build takes ~2-3 minutes.

That's it — you'll get a shareable `.streamlit.app` URL. Pushing new commits to
GitHub auto-redeploys.

## Updating the live app later

```powershell
git add .
git commit -m "describe your change"
git push
```

## Security reminders

- The key you pasted earlier into `.env.example` should be **revoked** and
  replaced. Use the new key only in `.env` (local) or the Secrets box (cloud).
- Never paste a real key into any `*.example` file or any file that gets
  committed.
