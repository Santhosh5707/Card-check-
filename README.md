# Card Check — dual screening prototype

This project has two separate workflows:
1. **Card image screening:** basic, explainable image-quality heuristics. This is not a trained counterfeit-card detector and cannot prove a card is genuine.
2. **Transaction fraud prediction:** a scikit-learn model trained from a labeled transaction dataset. Predictions stay disabled until a model is trained and installed; the app never fabricates predictions.

## Limitations and privacy
- A card photo cannot establish whether a card is active, stolen, cloned, or belongs to the presenter.
- For real verification, use an authorized payment processor/issuer integration.
- Do not upload real card photos or transaction data containing card numbers, CVVs, expiry dates, names, or other personal data.
- A real image ML model requires an authorized labeled dataset of legitimate and counterfeit/tampered card images and independent evaluation. No such model is included here.
- Transaction training expects a CSV with binary target column `Class` (1=fraud, 0=legitimate), such as a public anonymized dataset. Do not use sensitive customer data without authorization.

## Run locally (Python 3.10+)
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Open http://127.0.0.1:8000

## Train transaction model
Place an authorized dataset at `data/transactions.csv`, with target column `Class`, then run:
```bash
python train_transaction_model.py
```
This creates `models/transaction_model.joblib` and `models/metrics.json`. Restart the server. Review ROC-AUC and average precision; holdout metrics do not guarantee live performance, especially with imbalanced fraud data.

## Deployment
This is a FastAPI application and needs a Python-capable host. It is not a static-only Vercel site. Deploy to a host that supports Python dependencies and model files, or adapt and test it for serverless runtime limits. Never put secrets in frontend code.


## Train the card-image model (requires labeled images)

Install TensorFlow in Google Colab or a compatible Python environment:

```bash
pip install tensorflow
```

Organize images you have permission to use as:

```text
data/card_images/
  train/
    genuine/
    counterfeit/
  validation/
    genuine/
    counterfeit/
```

Then run `python train_image_model.py`. The script uses MobileNetV2 transfer learning and saves `models/card_image_model.keras`. The API uses this model automatically when that file exists.

**Critical:** you must have trustworthy labels for genuine and counterfeit/altered card images. Public synthetic card-image datasets are mainly for locating/detecting card objects, not proving counterfeit status. Do not label ordinary images as counterfeit just to make training run. If no suitable labeled image dataset is available, keep image classification marked as unavailable rather than showing invented predictions.
