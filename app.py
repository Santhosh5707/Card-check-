from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageStat, ImageFilter
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "transaction_model.joblib"
app = FastAPI(title="Card Check API", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")

class TransactionRequest(BaseModel):
    transactions: list[dict]

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

@app.get("/api/status")
def status():
    return {"image_screening": "basic visual heuristics available", "transaction_model_ready": MODEL_PATH.exists(),
            "transaction_model_note": "Trained model installed." if MODEL_PATH.exists() else "No trained transaction model installed. Follow README.md to train one."}

@app.post("/api/screen-image")
async def screen_image(file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPG, PNG, or WebP image.")
    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "Image must be 10 MB or smaller.")
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(400, "The uploaded file could not be read as an image.")
    # If a trained image classifier exists, use it; otherwise report only quality heuristics.
    image_model_path = ROOT / "models" / "card_image_model.keras"
    if image_model_path.exists():
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(image_model_path)
            prepared = image.resize((224, 224)).convert("RGB")
            batch = np.asarray(prepared, dtype=np.float32)[None, ...]
            p_counterfeit = float(model.predict(batch, verbose=0).reshape(-1)[0])
            label = "Potentially counterfeit / altered" if p_counterfeit >= 0.5 else "Model classified as likely genuine"
            return {
                "outcome": label,
                "explanation": "Prediction from the trained image classifier. This is a screening result, not proof of authenticity.",
                "classification": "TRAINED_IMAGE_MODEL_SCREEN — NOT ISSUER VERIFICATION",
                "probability_counterfeit": round(p_counterfeit, 6),
                "confidence_percent_for_reported_class": round((p_counterfeit if p_counterfeit >= 0.5 else 1-p_counterfeit) * 100, 2),
                "checks": [
                    {"name": "Trained image classifier", "status": "warning" if p_counterfeit >= 0.5 else "info",
                     "detail": f"Model-estimated counterfeit/alteration probability: {p_counterfeit*100:.2f}%"}
                ],
                "limitations": [
                    "The score is only as reliable as the labeled training data and independent evaluation.",
                    "A model cannot establish whether a card is active, stolen, cloned, or owned by the presenter.",
                    "Use an authorized issuer/payment-processor verification service for authoritative checks."
                ]
            }
        except ImportError:
            raise HTTPException(503, "A trained image model exists, but TensorFlow is not installed in this server runtime.")
        except Exception as exc:
            raise HTTPException(500, f"Could not run the trained image model: {exc}")
    w, h = image.size
    gray = image.convert("L")
    edge_var = float(np.var(np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32)))
    brightness = float(ImageStat.Stat(gray).mean[0])
    checks = [
        {"name":"Image size", "status":"pass" if min(w,h) >= 250 else "warning", "detail":f"{w} × {h} pixels"},
        {"name":"Sharpness proxy", "status":"pass" if edge_var >= 120 else "warning", "detail":f"Edge-variation score: {edge_var:.1f}"},
        {"name":"Brightness", "status":"pass" if 35 <= brightness <= 225 else "warning", "detail":f"Mean brightness: {brightness:.1f}/255"},
    ]
    warnings = sum(c["status"] == "warning" for c in checks)
    return {"outcome":"Image quality may be insufficient" if warnings >= 2 else "No major image-quality warning",
            "explanation":"Retake with even lighting, focus, and the full card visible." if warnings >= 2 else "This assesses general image properties only; it does not verify authenticity.",
            "classification":"HEURISTIC IMAGE SCREEN — NOT AUTHENTICITY VERIFICATION", "checks":checks,
            "limitations":["No trained counterfeit-card image model is included.", "These checks cannot determine whether a card is genuine, active, stolen, cloned, or owned by the presenter.", "Avoid images showing card numbers, CVV, or personal details."]}


@app.post("/api/predict-transactions")
def predict_transactions(payload: TransactionRequest):
    if not payload.transactions:
        raise HTTPException(400, "Provide at least one transaction row.")

    if len(payload.transactions) > 500:
        raise HTTPException(413, "Send at most 500 transactions per request.")

    if not MODEL_PATH.exists():
        raise HTTPException(503, "Transaction model is not installed.")

    try:
        model = joblib.load(MODEL_PATH)
        frame = pd.DataFrame(payload.transactions)

        # Get the exact feature names used during training.
        expected = getattr(model, "feature_names_in_", None)

        if expected is None and hasattr(model, "named_steps"):
            imputer = model.named_steps.get("imputer")
            expected = getattr(imputer, "feature_names_in_", None)

        if expected is None:
            raise HTTPException(
                500,
                "Could not determine the model's required feature columns."
            )

        expected = list(expected)
        missing = [c for c in expected if c not in frame.columns]

        if missing:
            raise HTTPException(
                400,
                f"Missing {len(missing)} required feature columns: "
                + ", ".join(missing)
            )

        # Use the same feature order as the training data.
        frame = frame[expected]
        probabilities = model.predict_proba(frame)[:, 1]

        results = []
        for i, probability in enumerate(probabilities):
            p = float(probability)
            results.append({
                "row": i + 1,
                "fraud_probability": round(p, 6),
                "risk": (
                    "High model-estimated risk"
                    if p >= 0.5
                    else "Lower model-estimated risk"
                )
            })

        return {
            "model": "Trained transaction classifier",
            "threshold": 0.5,
            "results": results,
            "warning": (
                "Estimates only, not proof of fraud. "
                "This model was trained on a limited labeled dataset; "
                "its results may not generalize to real-world transactions."
            )
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            400,
            f"Could not score the provided data: {exc}"
        )

