"""
StayBuddy — AI Complaint Categorization Server
===============================================
SEPARATE from the main AI recommendation server.

Run with:
  cd STAYBUDDY-main/ml-notebooks/complaint_ai
  python app_api.py

Runs on: http://127.0.0.1:8001
Flutter calls: POST http://127.0.0.1:8001/categorize
"""

import os
import re
import json
import joblib
import numpy as np
from scipy.sparse import hstack

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    os.system("pip install fastapi uvicorn pydantic scipy scikit-learn")
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load Models ────────────────────────────────────────────────────────────────
print("Loading Complaint AI models...")

try:
    model     = joblib.load(os.path.join(BASE_DIR, 'complaint_model.pkl'))
    word_vec  = joblib.load(os.path.join(BASE_DIR, 'complaint_word_vec.pkl'))
    char_vec  = joblib.load(os.path.join(BASE_DIR, 'complaint_char_vec.pkl'))
    label_enc = joblib.load(os.path.join(BASE_DIR, 'complaint_label_enc.pkl'))
    pri_model = joblib.load(os.path.join(BASE_DIR, 'complaint_priority_model.pkl'))
    pri_vec   = joblib.load(os.path.join(BASE_DIR, 'complaint_priority_vec.pkl'))

    with open(os.path.join(BASE_DIR, 'complaint_model_meta.json')) as f:
        meta = json.load(f)

    MODEL_READY = True
    print(f"✅ Complaint AI loaded")
    print(f"   Model     : {meta.get('model', 'Logistic Regression')}")
    print(f"   Accuracy  : {meta['accuracy']*100:.1f}%")
    print(f"   Categories: {len(meta['categories'])}")

except Exception as e:
    MODEL_READY = False
    print(f"❌ Failed to load models: {e}")
    print("   Make sure all .pkl files are in the same folder as this script.")
    print("   Run train_complaint_model.py first if .pkl files are missing.")


# ── Text Preprocessing ─────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Domain-specific expansions
    expansions = {
        'wifi':      'wifi internet connection',
        'internet':  'internet wifi connection',
        'warden':    'warden staff management',
        'cctv':      'cctv camera security surveillance',
        'stolen':    'stolen theft missing',
        'cockroach': 'cockroach pest insect',
        'smell':     'smell odor dirty unhygienic',
        'ac':        'air conditioning cooling',
    }
    for k, v in expansions.items():
        if f' {k} ' in f' {text} ':
            text = text.replace(k, v)
    return text


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    t  = preprocess(text)
    xw = word_vec.transform([t])
    xc = char_vec.transform([t])
    xv = hstack([xw, xc])

    # Category
    pred  = model.predict(xv)[0]
    proba = model.predict_proba(xv)[0]
    cat   = label_enc.inverse_transform([pred])[0]
    conf  = round(float(proba.max()), 3)

    # Top 3 categories
    top3 = [
        {
            'category':   label_enc.classes_[i],
            'confidence': round(float(proba[i]), 3)
        }
        for i in proba.argsort()[-3:][::-1]
    ]

    # Priority
    xp       = pri_vec.transform([t])
    priority = pri_model.predict(xp)[0]

    # Suggestion & subcategory from metadata
    suggestions   = meta['suggestions'].get(cat, ['Review and respond within 48 hours.'])
    subcategories = meta['subcategories'].get(cat, ['General'])

    # Pick most relevant suggestion
    best_sug = suggestions[0]
    for s in suggestions:
        if any(w in t for w in s.lower().split()[:3]):
            best_sug = s
            break

    return {
        'category':        cat,
        'subcategory':     subcategories[0] if subcategories else 'General',
        'priority':        priority,
        'confidence':      conf,
        'suggestion':      best_sug,
        'all_suggestions': suggestions,
        'top3_categories': top3,
    }


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="StayBuddy Complaint AI",
    description="AI-powered complaint categorization and suggestion engine",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Models ─────────────────────────────────────────────────────────────
class CategorizeRequest(BaseModel):
    text: str

class PatternRequest(BaseModel):
    complaints: list = []


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service":    "StayBuddy Complaint AI",
        "status":     "running" if MODEL_READY else "model not loaded",
        "accuracy":   f"{meta['accuracy']*100:.1f}%" if MODEL_READY else "N/A",
        "categories": meta['categories'] if MODEL_READY else [],
        "endpoints":  ["/categorize", "/pattern", "/health"]
    }


@app.get("/health")
def health():
    return {
        "status":      "ok" if MODEL_READY else "error",
        "model_ready": MODEL_READY,
        "accuracy":    meta.get('accuracy', 0) if MODEL_READY else 0,
        "model_type":  meta.get('model', '') if MODEL_READY else '',
        "n_categories": len(meta['categories']) if MODEL_READY else 0,
    }


@app.post("/categorize")
def categorize(req: CategorizeRequest):
    """
    Categorize a complaint text using the trained ML model.

    Returns:
    - category: Main complaint category
    - subcategory: Specific subcategory
    - priority: High / Medium / Low
    - confidence: Model confidence (0-1)
    - suggestion: AI-suggested action
    - all_suggestions: All suggestions for this category
    - top3_categories: Top 3 predicted categories with confidence
    """
    if not MODEL_READY:
        return {
            "success": False,
            "error": "Model not loaded. Make sure .pkl files are in the same folder."
        }
    if not req.text or not req.text.strip():
        return {
            "success": False,
            "error": "Please provide complaint text."
        }
    try:
        result = predict(req.text.strip())
        return {"success": True, **result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/pattern")
def pattern(req: PatternRequest):
    """
    Detect patterns in a list of complaints.
    Returns alert if any category appears 3+ times.

    Input: list of {"category": "...", "text": "..."}
    """
    try:
        from collections import Counter
        cats   = Counter(c.get("category", "Other") for c in req.complaints)
        alerts = []

        for cat, count in cats.items():
            if count >= 3:
                sug = (meta['suggestions'].get(cat, ['Review this issue systematically.'])[0]
                       if MODEL_READY else 'Review this issue systematically.')
                alerts.append({
                    "category": cat,
                    "count":    count,
                    "severity": "High" if count >= 5 else "Medium",
                    "message":  f"{count} '{cat}' complaints — possible systemic issue",
                    "suggestion": sug,
                })

        alerts.sort(key=lambda x: -x['count'])
        return {
            "success": True,
            "total_complaints": len(req.complaints),
            "patterns_found":   len(alerts),
            "alerts":           alerts,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "alerts": []}


@app.post("/batch")
def batch_categorize(complaints: list):
    """
    Categorize multiple complaints at once.
    Input: ["complaint text 1", "complaint text 2", ...]
    """
    if not MODEL_READY:
        return {"success": False, "error": "Model not loaded."}
    try:
        results = []
        for text in complaints:
            r = predict(str(text))
            results.append({"text": text, **r})
        return {"success": True, "count": len(results), "results": results}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    print()
    print("=" * 55)
    print("  StayBuddy Complaint AI Server")
    print("  Running on: http://127.0.0.1:8001")
    print("  Docs:       http://127.0.0.1:8001/docs")
    print()
    print("  Flutter calls:")
    print("    POST http://127.0.0.1:8001/categorize")
    print("    POST http://127.0.0.1:8001/pattern")
    print("=" * 55)
    print()
    uvicorn.run(app, host="127.0.0.1", port=8001, reload=False)
