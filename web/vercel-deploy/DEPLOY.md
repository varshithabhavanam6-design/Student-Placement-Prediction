# Deploying to Vercel

This folder is a **static site** — just `index.html` + `student_placement.csv`.
No build step, no `package.json`, no server. Vercel deploys this in about a minute.

## Option A — GitHub (recommended, auto-redeploys on every push)

1. Create a new GitHub repo (or add a `web/` folder to your existing
   `Student-Placement-Prediction` repo) and push these two files:
   - `index.html`
   - `student_placement.csv`
2. Go to https://vercel.com → **Add New → Project**.
3. Import that GitHub repo.
4. Framework preset: leave it as **Other** (Vercel auto-detects static HTML).
   - If you put these files in a subfolder like `web/`, set
     **Root Directory** to `web` in the import settings.
5. Click **Deploy**. You'll get a live URL like
   `https://student-placement-predictor.vercel.app` within ~60 seconds.

## Option B — Vercel CLI (no GitHub needed)

```bash
npm install -g vercel
cd vercel-deploy        # the folder containing index.html
vercel                  # follow the prompts, accept defaults
vercel --prod           # promote to your production URL
```

## After deploying

- Every push to your connected GitHub branch auto-redeploys.
- Add the live link to your resume/LinkedIn alongside your Streamlit Cloud link —
  having both shows you can ship to multiple hosting models.
- This page runs the **real trained Logistic Regression weights** baked in as
  JavaScript — verified to match scikit-learn's `predict_proba()` output. If you
  ever retrain the model with new data, re-export `coef_`, `intercept_`,
  `scaler.mean_`, and `scaler.scale_` from Python and swap the constants in
  `index.html`'s `<script>` block.
