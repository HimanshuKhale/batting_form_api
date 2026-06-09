"""
Khel AI - Batting Form Flask API

Deployable on Render with:
  gunicorn app:app

Artifacts expected in ./artifacts:
  - khel_ai_batting_form_model.pkl
  - batting_form_ball_by_ball.csv
  - player_form_index.csv
"""

from __future__ import annotations

import os
import math
import traceback
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE_DIR / "artifacts"))
MODEL_PATH = ARTIFACT_DIR / "khel_ai_batting_form_model.pkl"
PLAYER_FORM_PATH = ARTIFACT_DIR / "player_form_index.csv"
BALL_FORM_PATH = ARTIFACT_DIR / "batting_form_ball_by_ball.csv"

app = Flask(__name__)
CORS(app)

_model_package: Dict[str, Any] | None = None
_player_df: pd.DataFrame | None = None
_ball_df: pd.DataFrame | None = None


def _clean_value(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def _clean_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    return {str(k): _clean_value(v) for k, v in row.items()}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def form_label(score: float) -> str:
    """Score is expected from 0 to 1."""
    score = float(score)
    if score >= 0.85:
        return "Elite Form"
    if score >= 0.70:
        return "Good Form"
    if score >= 0.50:
        return "Stable Form"
    if score >= 0.30:
        return "Struggling Form"
    return "Poor Form"


def _load_csv_artifacts() -> None:
    """Load CSV artifacts before loading/training the model."""
    global _player_df, _ball_df

    if _player_df is None:
        if not PLAYER_FORM_PATH.exists():
            raise FileNotFoundError(f"Player form CSV not found: {PLAYER_FORM_PATH}")
        _player_df = pd.read_csv(PLAYER_FORM_PATH)

    if _ball_df is None:
        if not BALL_FORM_PATH.exists():
            raise FileNotFoundError(f"Ball-by-ball form CSV not found: {BALL_FORM_PATH}")
        _ball_df = pd.read_csv(BALL_FORM_PATH)


def _train_fallback_model(load_error: Exception | None = None) -> Dict[str, Any]:
    """
    Train a lightweight replacement model from the included CSV if the PKL cannot be loaded.

    Why this exists:
    Pickle/joblib model files can break when Colab and Render use different
    NumPy / scikit-learn versions. The CSV is portable, so this fallback keeps
    the API online even when the serialized PKL is incompatible.
    """
    _load_csv_artifacts()
    assert _ball_df is not None

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    feature_columns = [
        "runs_off_bat",
        "extras",
        "current_run_rate",
        "required_run_rate",
        "pressure_index",
        "expected_runs",
        "expected_wicket_probability",
        "shot_risk_score",
        "batter_control_score",
        "target_is_wicket",
        "target_total_runs",
    ]

    train_df = _ball_df.copy()
    if "target_total_runs" not in train_df.columns:
        train_df["target_total_runs"] = train_df.get("runs_off_bat", 0).fillna(0) + train_df.get("extras", 0).fillna(0)

    target_col = "batting_form_index"
    if target_col not in train_df.columns:
        raise ValueError(f"Cannot train fallback model because {target_col} is missing from batting_form_ball_by_ball.csv")

    for col in feature_columns:
        if col not in train_df.columns:
            train_df[col] = 0

    model_df = train_df[feature_columns + [target_col]].copy()
    model_df = model_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = model_df[feature_columns]
    y = model_df[target_col].clip(0, 1)

    if len(model_df) >= 20:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = GradientBoostingRegressor(random_state=42, n_estimators=120, max_depth=3, learning_rate=0.05)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": round(float(mean_absolute_error(y_test, preds)), 5),
        "r2": round(float(r2_score(y_test, preds)), 5) if len(y_test) > 1 else None,
        "fallback_trained_rows": int(len(model_df)),
    }

    return {
        "model": model,
        "model_name": "FallbackGradientBoostingRegressor",
        "feature_columns": feature_columns,
        "target": target_col,
        "metrics": metrics,
        "description": "Fallback model trained at startup from batting_form_ball_by_ball.csv because the uploaded PKL could not be loaded safely.",
        "scale": "0 to 1 normalized batting form index",
        "form_scale_interpretation": {
            "0.00-0.30": "Poor Form",
            "0.30-0.50": "Struggling Form",
            "0.50-0.70": "Stable Form",
            "0.70-0.85": "Good Form",
            "0.85-1.00": "Elite Form",
        },
        "pkl_load_error": str(load_error) if load_error else None,
    }


def load_artifacts() -> None:
    """Lazy-load model and CSV files once per process."""
    global _model_package, _player_df, _ball_df

    _load_csv_artifacts()

    if _model_package is None:
        if MODEL_PATH.exists():
            try:
                _model_package = joblib.load(MODEL_PATH)
            except Exception as exc:
                app.logger.warning("Could not load PKL model, training fallback model. Error: %s", exc)
                app.logger.debug(traceback.format_exc())
                _model_package = _train_fallback_model(load_error=exc)
        else:
            app.logger.warning("Model file missing, training fallback model from CSV: %s", MODEL_PATH)
            _model_package = _train_fallback_model(load_error=FileNotFoundError(str(MODEL_PATH)))


def model_package() -> Dict[str, Any]:
    load_artifacts()
    assert _model_package is not None
    if isinstance(_model_package, dict):
        return _model_package
    return {"model": _model_package, "feature_columns": []}


def player_df() -> pd.DataFrame:
    load_artifacts()
    assert _player_df is not None
    return _player_df.copy()


def ball_df() -> pd.DataFrame:
    load_artifacts()
    assert _ball_df is not None
    return _ball_df.copy()


def build_feature_row(payload: Dict[str, Any]) -> Dict[str, float]:
    """Build the exact feature row expected by the saved model."""
    runs = to_float(payload.get("runs_off_bat"), 0)
    extras = to_float(payload.get("extras"), 0)

    row = {
        "runs_off_bat": runs,
        "extras": extras,
        "current_run_rate": to_float(payload.get("current_run_rate"), 0),
        "required_run_rate": to_float(payload.get("required_run_rate"), 0),
        "pressure_index": to_float(payload.get("pressure_index"), 0),
        "expected_runs": to_float(payload.get("expected_runs"), 1.2),
        "expected_wicket_probability": to_float(payload.get("expected_wicket_probability"), 0.03),
        "shot_risk_score": to_float(payload.get("shot_risk_score"), 0.5),
        "batter_control_score": to_float(payload.get("batter_control_score"), 0.5),
        "target_is_wicket": to_int(payload.get("target_is_wicket"), 0),
        # In the notebook this feature represented total runs outcome for the ball.
        # If not supplied, default to runs + extras.
        "target_total_runs": to_float(payload.get("target_total_runs"), runs + extras),
    }
    return row


def predict_form_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    pkg = model_package()
    model = pkg.get("model")
    feature_columns = pkg.get("feature_columns") or list(build_feature_row({}).keys())

    feature_row = build_feature_row(payload)

    for col in feature_columns:
        feature_row.setdefault(col, 0.0)

    X = pd.DataFrame([feature_row])[feature_columns].fillna(0)
    raw_prediction = float(model.predict(X)[0])
    normalized_prediction = clamp(raw_prediction, 0.0, 1.0)

    return {
        "batting_form_index": round(normalized_prediction, 4),
        "batting_form_score_100": round(normalized_prediction * 100, 2),
        "form_label": form_label(normalized_prediction),
        "model_name": pkg.get("model_name", type(model).__name__),
        "target": pkg.get("target", "batting_form_index"),
        "feature_columns": feature_columns,
        "input_features": _clean_dict(feature_row),
    }


def latest_player_profile(striker_id: int) -> Dict[str, Any] | None:
    df = player_df()
    if "striker_id" not in df.columns:
        return None
    row_df = df[df["striker_id"] == int(striker_id)]
    if row_df.empty:
        return None
    row = _clean_dict(row_df.iloc[0].to_dict())
    latest = to_float(row.get("latest_form_index"), 0)
    row["latest_form_score_100"] = round(latest * 100, 2)
    row["form_label"] = form_label(latest)
    return row


@app.errorhandler(Exception)
def handle_exception(exc: Exception):
    # Keep API responses clean instead of returning HTML error pages.
    return jsonify({"error": str(exc), "type": type(exc).__name__}), 500


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health")
def health():
    try:
        load_artifacts()
        pkg = model_package()
        return jsonify({
            "status": "ok",
            "service": "Khel AI Batting Form API",
            "artifact_dir": str(ARTIFACT_DIR),
            "model_loaded": True,
            "model_name": pkg.get("model_name", type(pkg.get("model")).__name__),
            "model_source": "fallback_trained_from_csv" if pkg.get("pkl_load_error") else "pkl",
            "pkl_load_error": pkg.get("pkl_load_error"),
            "player_rows": int(len(player_df())),
            "ball_rows": int(len(ball_df())),
        })
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/api/model-info")
def api_model_info():
    pkg = model_package()
    model = pkg.get("model")
    feature_importance = []
    columns = pkg.get("feature_columns") or []
    if hasattr(model, "feature_importances_") and columns:
        feature_importance = sorted(
            [
                {"feature": col, "importance": round(float(imp), 5)}
                for col, imp in zip(columns, model.feature_importances_)
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )

    return jsonify({
        "model_name": pkg.get("model_name", type(model).__name__),
        "target": pkg.get("target"),
        "description": pkg.get("description"),
        "scale": pkg.get("scale"),
        "form_scale_interpretation": pkg.get("form_scale_interpretation"),
        "feature_columns": columns,
        "metrics": pkg.get("metrics"),
        "feature_importance": feature_importance,
    })


@app.route("/api/players")
def api_players():
    df = player_df()
    search = request.args.get("search", "").strip().lower()
    limit = to_int(request.args.get("limit"), 50)

    if search and "batter_name" in df.columns:
        df = df[df["batter_name"].astype(str).str.lower().str.contains(search, na=False)]

    df = df.sort_values("latest_form_index", ascending=False).head(limit)
    records = []
    for _, row in df.iterrows():
        item = _clean_dict(row.to_dict())
        latest = to_float(item.get("latest_form_index"), 0)
        item["latest_form_score_100"] = round(latest * 100, 2)
        item["form_label"] = form_label(latest)
        records.append(item)

    return jsonify({"count": len(records), "results": records})


@app.route("/api/leaderboard")
def api_leaderboard():
    df = player_df()
    metric = request.args.get("metric", "latest_form_index")
    limit = to_int(request.args.get("limit"), 10)

    allowed_metrics = {
        "latest_form_index",
        "best_form_index",
        "avg_ball_impact",
        "total_runs",
        "balls_faced",
    }
    if metric not in allowed_metrics:
        return jsonify({"error": f"Invalid metric. Use one of: {sorted(allowed_metrics)}"}), 400

    df = df.sort_values(metric, ascending=False).head(limit)
    results = []
    for _, row in df.iterrows():
        item = _clean_dict(row.to_dict())
        latest = to_float(item.get("latest_form_index"), 0)
        item["latest_form_score_100"] = round(latest * 100, 2)
        item["form_label"] = form_label(latest)
        results.append(item)
    return jsonify({"metric": metric, "count": len(results), "results": results})


@app.route("/api/player/<int:striker_id>")
def api_player_profile(striker_id: int):
    profile = latest_player_profile(striker_id)
    if profile is None:
        return jsonify({"error": f"No player found with striker_id={striker_id}"}), 404

    history_limit = to_int(request.args.get("history_limit"), 20)
    history = ball_df()
    if "striker_id" in history.columns:
        history = history[history["striker_id"] == striker_id]
        history = history.sort_values(["innings_id", "over_number", "ball_number", "ball_event_id"]).tail(history_limit)
        profile["recent_balls"] = [_clean_dict(r) for r in history.to_dict(orient="records")]

    return jsonify(profile)


@app.route("/api/player/<int:striker_id>/history")
def api_player_history(striker_id: int):
    limit = to_int(request.args.get("limit"), 100)
    df = ball_df()
    if "striker_id" not in df.columns:
        return jsonify({"error": "striker_id column missing in ball dataset"}), 500
    df = df[df["striker_id"] == striker_id]
    df = df.sort_values(["innings_id", "over_number", "ball_number", "ball_event_id"]).tail(limit)
    records = [_clean_dict(r) for r in df.to_dict(orient="records")]
    return jsonify({"striker_id": striker_id, "count": len(records), "results": records})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    result = predict_form_from_payload(payload)
    return jsonify(result)


@app.route("/api/simulate", methods=["POST"])
def api_simulate():
    """
    Same as /api/predict, with optional previous_form_index to show updated form movement.
    This is useful for ball-by-ball dashboard simulation.
    """
    payload = request.get_json(silent=True) or {}
    prediction = predict_form_from_payload(payload)
    previous_form = to_float(payload.get("previous_form_index"), prediction["batting_form_index"])
    memory = clamp(to_float(payload.get("memory"), 0.85), 0.0, 0.99)
    predicted_ball_impact = prediction["batting_form_index"]
    updated_form = (memory * previous_form) + ((1 - memory) * predicted_ball_impact)
    updated_form = clamp(updated_form, 0.0, 1.0)

    prediction.update({
        "previous_form_index": round(previous_form, 4),
        "memory": round(memory, 3),
        "updated_form_index": round(updated_form, 4),
        "updated_form_score_100": round(updated_form * 100, 2),
        "updated_form_label": form_label(updated_form),
    })
    return jsonify(prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
