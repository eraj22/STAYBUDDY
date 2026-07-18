"""
StayBuddy — AI Recommendation API Bridge
Exposes the hybrid ML engine as a REST endpoint for the Flutter app.

Run with:
  cd STAYBUDDY-main/ml-notebooks
  pip install -r requirements.txt
  python app_api.py

Runs on: http://127.0.0.1:8000
Flutter calls: POST /recommend
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Installing required packages...")
    os.system("pip install fastapi uvicorn pydantic")
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load data & models once at startup ────────────────────────────
print("Loading dataset and models...")

hostels_df      = pd.read_csv(os.path.join(DATA_DIR, "hostels.csv"))
students_df     = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
interactions_df = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])

hostel_matrix      = np.load(os.path.join(MODEL_DIR, "hostel_feature_matrix.npy"))
predicted_matrix   = pd.read_csv(os.path.join(MODEL_DIR, "predicted_matrix.csv"), index_col=0)
cold_start_models  = joblib.load(os.path.join(MODEL_DIR, "cold_start_models.pkl"))

with open(os.path.join(MODEL_DIR, "hybrid_config.json")) as f:
    hybrid_config = json.load(f)

best_alpha  = hybrid_config["best_alpha"]
type_alphas = hybrid_config["type_alphas"]

print(f"✅ Loaded {len(hostels_df)} hostels, {len(students_df)} students")
print(f"✅ Hybrid α = {best_alpha}")

# ── Constants ──────────────────────────────────────────────────────
TECH_DEPTS = [
    "Computer Science", "Electrical Engineering",
    "Software Engineering", "Cyber Security", "Data Science"
]

# ── Helper functions (same logic as app.py) ────────────────────────
def classify_student_type(study_pref, price_sens, comfort_pref):
    if study_pref > 0.70:   return "study_focused"
    if price_sens > 0.70:   return "budget_conscious"
    if comfort_pref > 0.70: return "comfort_seeking"
    return "balanced"

def food_compat(student_pref, hostel_food):
    if hostel_food == "None":       return 0.8
    if student_pref == "Both":      return 1.0
    if student_pref == hostel_food: return 1.0
    if hostel_food == "Both":       return 0.9
    return 0.3

def room_compat(preferred, available_json):
    try:
        available = json.loads(available_json)
    except:
        return 0.5
    if preferred in available: return 1.0
    alts = {"Single": ["Double"], "Double": ["Single", "Triple"],
            "Dormitory": ["Triple"], "Triple": ["Double", "Dormitory"]}
    return 0.6 if any(a in available for a in alts.get(preferred, [])) else 0.3

def build_student_vector(gender, department, budget_max, max_dist,
                          study_pref, food_pref, room_type, price_sens,
                          comfort_pref, noise_tol, curfew_flex,
                          needs_transport, must_have):
    price_score    = np.clip(1 - (budget_max / hostels_df["single_room_price"].max()), 0, 1)
    dist_score     = np.clip(1 - (max_dist / hostels_df["distance_from_fast_km"].max()), 0, 1)
    safety_score   = 0.90 if gender == "Female" else 0.70
    food_map       = {"Veg": 0.33, "Non-Veg": 0.66, "Both": 1.0}
    food_score     = food_map.get(food_pref, 0.5)
    internet_score = 0.90 if department in TECH_DEPTS else 0.50
    amenity_score  = min(len(must_have) / 14, 1.0)

    base = np.array([
        price_score, dist_score, study_pref, study_pref,
        safety_score, comfort_pref, amenity_score, price_sens,
        internet_score, 1 - noise_tol, food_score, curfew_flex,
        float(needs_transport), 1.0 if food_pref != "None" else 0.0,
        price_sens * 0.50
    ])
    weights = np.array([
        price_sens, 1.0, study_pref, study_pref,
        0.90 if gender == "Female" else 0.70,
        comfort_pref, min(len(must_have) / 5, 1.0), price_sens,
        internet_score, 1 - noise_tol, 0.80,
        curfew_flex, float(needs_transport), 0.70, price_sens * 0.50
    ])
    weights = weights / (weights.sum() + 1e-9)
    return base * weights

def get_recommendations(gender, department, budget_max, max_dist,
                         study_pref, food_pref, room_type, price_sens,
                         comfort_pref, noise_tol, curfew_flex,
                         needs_transport, must_have, top_k=5):

    hostel_type = "Girls" if gender == "Female" else "Boys"
    gender_mask = hostels_df["hostel_type"] == hostel_type
    filt_h      = hostels_df[gender_mask].copy()
    filt_m      = hostel_matrix[gender_mask.values]

    # Content-Based scores
    svec  = build_student_vector(
        gender, department, budget_max, max_dist, study_pref,
        food_pref, room_type, price_sens, comfort_pref,
        noise_tol, curfew_flex, needs_transport, must_have
    ).reshape(1, -1)
    sims  = cosine_similarity(svec, filt_m)[0]
    sims *= filt_h["food_type"].apply(lambda ft: food_compat(food_pref, ft)).values
    sims *= (0.7 + 0.3 * filt_h["room_types_available"].apply(
        lambda rt: room_compat(room_type, rt)).values)
    sims *= (0.85 + 0.15 * (filt_h["available_rooms"] > 0).astype(float).values)

    cb_scores = pd.Series(sims, index=filt_h["hostel_id"].values)
    if cb_scores.max() > cb_scores.min():
        cb_scores = (cb_scores - cb_scores.min()) / (cb_scores.max() - cb_scores.min())

    # Collaborative scores via cold-start clustering
    model = cold_start_models.get(gender)
    if model is not None:
        feat_vec = np.array([[
            budget_max, max_dist, study_pref, price_sens, comfort_pref, noise_tol,
            1 if "WiFi" in must_have else 0,
            1 if "Study Room" in must_have else 0,
            1 if "AC" in must_have else 0,
            1 if "Generator" in must_have else 0,
            int(needs_transport)
        ]])
        scaled  = model["scaler"].transform(feat_vec)
        cluster = model["kmeans"].predict(scaled)[0]
        labels  = model["kmeans"].labels_
        sim_idx = [model["indices"][i] for i, lbl in enumerate(labels) if lbl == cluster]
        sim_sids = students_df.loc[sim_idx, "student_id"].tolist()
        avg = predicted_matrix.loc[predicted_matrix.index.isin(sim_sids)].mean(axis=0)
        cf_scores = avg[avg.index.isin(filt_h["hostel_id"].values)]
    else:
        cf_scores = filt_h.set_index("hostel_id")["overall_rating"] / 5

    if cf_scores.max() > cf_scores.min():
        cf_scores = (cf_scores - cf_scores.min()) / (cf_scores.max() - cf_scores.min())

    # Hybrid fusion
    stype = classify_student_type(study_pref, price_sens, comfort_pref)
    alpha = type_alphas.get(stype, best_alpha)
    idx   = cb_scores.index.union(cf_scores.index)
    hybrid = (
        alpha * cb_scores.reindex(idx, fill_value=0) +
        (1 - alpha) * cf_scores.reindex(idx, fill_value=0)
    )
    if hybrid.max() > hybrid.min():
        hybrid = (hybrid - hybrid.min()) / (hybrid.max() - hybrid.min())

    top_ids = hybrid.nlargest(top_k).index.tolist()
    results = filt_h[filt_h["hostel_id"].isin(top_ids)].copy()
    results["hybrid_score"] = results["hostel_id"].map(hybrid)
    results["cb_score"]     = results["hostel_id"].map(
        cb_scores.reindex(results["hostel_id"], fill_value=0))
    results["cf_score"]     = results["hostel_id"].map(
        cf_scores.reindex(results["hostel_id"], fill_value=0))
    results["alpha_used"]   = alpha
    results["student_type"] = stype

    return results.sort_values("hybrid_score", ascending=False).reset_index(drop=True)


# ── FastAPI app ────────────────────────────────────────────────────
app = FastAPI(title="StayBuddy AI API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendRequest(BaseModel):
    gender: str = "Female"
    department: str = "Computer Science"
    budget_max: int = 20000
    max_distance_km: float = 3.0
    study_preference: float = 0.6
    food_preference: str = "Both"
    room_type: str = "Single"
    price_sensitivity: float = 0.6
    comfort_preference: float = 0.5
    noise_tolerance: float = 0.3
    curfew_flexibility: float = 0.5
    needs_transport: bool = False
    must_have: list = ["WiFi", "Hot Water"]
    top_k: int = 5

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "StayBuddy AI Recommendation Engine",
        "endpoints": ["/recommend", "/health"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "hostels_loaded": len(hostels_df),
        "students_loaded": len(students_df),
        "best_alpha": best_alpha
    }

@app.post("/recommend")
def recommend(req: RecommendRequest):
    try:
        results = get_recommendations(
            gender=req.gender,
            department=req.department,
            budget_max=req.budget_max,
            max_dist=req.max_distance_km,
            study_pref=req.study_preference,
            food_pref=req.food_preference,
            room_type=req.room_type,
            price_sens=req.price_sensitivity,
            comfort_pref=req.comfort_preference,
            noise_tol=req.noise_tolerance,
            curfew_flex=req.curfew_flexibility,
            needs_transport=req.needs_transport,
            must_have=req.must_have,
            top_k=req.top_k,
        )

        recs = []
        for _, row in results.iterrows():
            recs.append({
                "hostel_id":             str(row["hostel_id"]),
                "hostel_name":           str(row["hostel_name"]),
                "hostel_type":           str(row["hostel_type"]),
                "area":                  str(row["area"]),
                "latitude":              float(row["latitude"]),
                "longitude":             float(row["longitude"]),
                "single_room_price":     float(row["single_room_price"]),
                "distance_from_fast_km": float(row["distance_from_fast_km"]),
                "overall_rating":        float(row["overall_rating"]),
                "hybrid_score":          round(float(row["hybrid_score"]), 4),
                "cb_score":              round(float(row["cb_score"]), 4),
                "cf_score":              round(float(row["cf_score"]), 4),
                "student_type":          str(row["student_type"]),
                "alpha_used":            float(row["alpha_used"]),
            })

        return {
            "success": True,
            "student_type": results.iloc[0]["student_type"] if len(results) > 0 else "unknown",
            "alpha_used": float(results.iloc[0]["alpha_used"]) if len(results) > 0 else best_alpha,
            "count": len(recs),
            "recommendations": recs,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "recommendations": []}


# ── Chatbot endpoint ───────────────────────────────────────────────
_chatbot_state = None
_chatbot_error = None

def get_chatbot_state():
    global _chatbot_state, _chatbot_error
    if _chatbot_state is not None or _chatbot_error is not None:
        return _chatbot_state

    try:
        from chatbot import (
            load_intent_model, load_spacy,
            ConversationContext, chat, AMENITY_LABELS,
        )
        tokenizer, model, label_encoder = load_intent_model()
        _chatbot_state = {
            "tokenizer": tokenizer,
            "model": model,
            "label_encoder": label_encoder,
            "nlp": load_spacy(),
            "context": ConversationContext(),
            "chat": chat,
        }
        print("✅ Chatbot (DistilBERT) loaded")
    except Exception as e:
        _chatbot_error = str(e)
        print(f"⚠️  Chatbot not loaded: {_chatbot_error}")

    return _chatbot_state

class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    chatbot_state = get_chatbot_state()
    if chatbot_state is None:
        return {
            "response": "Chatbot model is not loaded. Make sure intent_model/ and label_encoder.pkl are present.",
            "intent": "error",
            "confidence": 0.0,
            "hostels": [],
        }
    try:
        response = chatbot_state["chat"](
            user_text  = req.message,
            context    = chatbot_state["context"],
            tokenizer  = chatbot_state["tokenizer"],
            model      = chatbot_state["model"],
            le         = chatbot_state["label_encoder"],
            nlp        = chatbot_state["nlp"],
            hostels_df = hostels_df,
            rec_fn     = get_recommendations,
        )

        # Extract hostel list if present
        hostels_out = []
        if response.get("type") == "hostel_results":
            for h in response.get("hostels", []):
                hostels_out.append({
                    "name":     h.get("name", ""),
                    "area":     h.get("area", ""),
                    "price":    h.get("price", 0),
                    "rating":   h.get("rating", 0),
                    "distance": h.get("distance", 0),
                    "score":    h.get("score"),
                })

        return {
            "response":   response.get("message", ""),
            "intent":     response.get("intent", ""),
            "confidence": float(response.get("confidence", 0.0)),
            "hostels":    hostels_out,
        }
    except Exception as e:
        return {
            "response":   f"Sorry, something went wrong: {str(e)}",
            "intent":     "error",
            "confidence": 0.0,
            "hostels":    [],
        }


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  StayBuddy AI Recommendation API")
    print("  Running on: http://127.0.0.1:8000")
    print("  Flutter calls: POST http://127.0.0.1:8000/recommend")
    print("="*55 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)