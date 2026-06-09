# Khel AI Batting Form Flask API

This is a Render-ready Flask API for the **Khel AI Batting Form Model** exported from the Colab notebook.

It uses these artifacts:

- `artifacts/khel_ai_batting_form_model.pkl`
- `artifacts/batting_form_ball_by_ball.csv`
- `artifacts/player_form_index.csv`
- `artifacts/khel_ai_batting_form_model_colab.ipynb`

## 1. Local setup

```bash
python -m venv venv
```

### Windows PowerShell

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run locally:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Health check:

```text
http://127.0.0.1:5000/health
```

## 2. Production run command

Render should use:

```bash
gunicorn app:app
```

## 3. API endpoints

### Home page

```http
GET /
```

A small browser test page.

### Health check

```http
GET /health
```

Checks whether the model and CSV files load correctly.

### Model info

```http
GET /api/model-info
```

Returns model name, feature columns, metrics, scale, interpretation, and feature importance if available.

### Players

```http
GET /api/players
GET /api/players?limit=20
GET /api/players?search=manan
```

Returns player form records from `player_form_index.csv`.

### Leaderboard

```http
GET /api/leaderboard
GET /api/leaderboard?metric=latest_form_index&limit=10
GET /api/leaderboard?metric=best_form_index&limit=10
```

Allowed metrics:

- `latest_form_index`
- `best_form_index`
- `avg_ball_impact`
- `total_runs`
- `balls_faced`

### Player profile

```http
GET /api/player/15
GET /api/player/15?history_limit=30
```

Returns one batter's profile and recent ball history.

### Player ball history

```http
GET /api/player/15/history
GET /api/player/15/history?limit=100
```

Returns ball-by-ball batting form history for one player.

### Predict batting form

```http
POST /api/predict
Content-Type: application/json
```

Sample body:

```json
{
  "runs_off_bat": 4,
  "extras": 0,
  "current_run_rate": 8.5,
  "required_run_rate": 9.2,
  "pressure_index": 0.72,
  "expected_runs": 1.4,
  "expected_wicket_probability": 0.08,
  "shot_risk_score": 0.45,
  "batter_control_score": 0.82,
  "target_is_wicket": 0,
  "target_total_runs": 4
}
```

### Simulate updated form after a ball

```http
POST /api/simulate
Content-Type: application/json
```

Sample body:

```json
{
  "previous_form_index": 0.62,
  "memory": 0.85,
  "runs_off_bat": 4,
  "extras": 0,
  "current_run_rate": 8.5,
  "required_run_rate": 9.2,
  "pressure_index": 0.72,
  "expected_runs": 1.4,
  "expected_wicket_probability": 0.08,
  "shot_risk_score": 0.45,
  "batter_control_score": 0.82,
  "target_is_wicket": 0,
  "target_total_runs": 4
}
```

## 4. Deploy on Render

### Step A — Push this folder to GitHub

Create a new GitHub repository, for example:

```text
khel-ai-batting-form-api
```

Then from this folder:

```bash
git init
git add .
git commit -m "Add Khel AI batting form Flask API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/khel-ai-batting-form-api.git
git push -u origin main
```

### Step B — Create Render Web Service

1. Go to Render Dashboard.
2. Click **New > Web Service**.
3. Connect the GitHub repository.
4. Use these settings:

```text
Language: Python 3
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

The included `render.yaml` also supports Render Blueprint deployment.

### Step C — Test after deploy

When Render gives you a URL like:

```text
https://khel-ai-batting-form-api.onrender.com
```

Open:

```text
https://khel-ai-batting-form-api.onrender.com/health
```

Then test:

```text
https://khel-ai-batting-form-api.onrender.com/api/leaderboard
```

## 5. Connect to Khel AI MVP frontend

From your Django template or JS file:

```html
<script>
async function loadBattingForm(playerId) {
  const baseUrl = "https://YOUR-RENDER-URL.onrender.com";
  const response = await fetch(`${baseUrl}/api/player/${playerId}`);
  const data = await response.json();

  console.log(data);

  // Example UI mapping:
  // document.getElementById("form-score").innerText = data.latest_form_score_100;
  // document.getElementById("form-label").innerText = data.form_label;
}

loadBattingForm(15);
</script>
```

For live ball prediction:

```js
async function predictBallForm() {
  const baseUrl = "https://YOUR-RENDER-URL.onrender.com";
  const payload = {
    runs_off_bat: 4,
    extras: 0,
    current_run_rate: 8.5,
    required_run_rate: 9.2,
    pressure_index: 0.72,
    expected_runs: 1.4,
    expected_wicket_probability: 0.08,
    shot_risk_score: 0.45,
    batter_control_score: 0.82,
    target_is_wicket: 0,
    target_total_runs: 4
  };

  const response = await fetch(`${baseUrl}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  console.log(data);
}
```

## 6. Notes

- The model was saved using scikit-learn 1.6.1, so `requirements.txt` pins `scikit-learn==1.6.1`.
- The API supports CORS so your Django app can call it from another domain.
- The PKL file is loaded once per worker process, not on every request.
- For larger future artifacts, store files in cloud storage or Git LFS instead of normal GitHub.

## Fix for NumPy / pickle BitGenerator error

If Render shows an error like:

```json
{"error": "<class 'numpy.random._mt19937.MT19937'> is not a known BitGenerator module.", "type": "ValueError"}
```

that means the `.pkl` was created in a different NumPy/scikit-learn environment than the one used on Render. This app now handles that safely:

1. It first tries to load `artifacts/khel_ai_batting_form_model.pkl`.
2. If the PKL fails, it automatically trains a fallback `GradientBoostingRegressor` from `artifacts/batting_form_ball_by_ball.csv`.
3. The `/health` endpoint will show whether the API is using the original PKL or the fallback model.

The pinned dependency versions in `requirements.txt` are also set to reduce pickle incompatibility.
