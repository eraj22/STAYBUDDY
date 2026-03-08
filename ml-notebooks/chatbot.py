"""
╔══════════════════════════════════════════════════════════════════╗
║   StayBuddy — NLP Chatbot Core                                   ║
║   Author  : Samiya Saleem (22I-1065)                             ║
║   Model   : DistilBERT fine-tuned on 390 intent examples         ║
║   Intents : 7  (hostel_search, amenity_inquiry, pricing_info,    ║
║              booking_process, location_info, complaint,          ║
║              general_info)                                        ║
╚══════════════════════════════════════════════════════════════════╝

INTELLIGENT features:
  1. DistilBERT transformer — not keyword matching
  2. Confidence threshold (< 0.65 → asks clarification)
  3. spaCy entity extraction (budget, amenity, room, distance)
  4. Multi-turn context — remembers gender, budget across messages
  5. hostel_search → fires Eraj's hybrid recommendation engine
  6. All other intents → query hostels.csv directly
"""

import re
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import spacy

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "intent_model"
DATA_DIR  = BASE_DIR / "data"

# ── Fallback: if path has spaces, use C:\staybuddy_models ──────────
_CLEAN_BASE = Path("C:/staybuddy_models")
if " " in str(BASE_DIR) and _CLEAN_BASE.exists():
    MODEL_DIR = _CLEAN_BASE / "intent_model"
    _LABEL_PATH = _CLEAN_BASE / "label_encoder.pkl"
else:
    _LABEL_PATH = BASE_DIR / "label_encoder.pkl"

# ── Constants ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.65   # below this → ask for clarification
MAX_RESULTS          = 5      # hostel results to return

AMENITY_MAP = {
    "wifi":             "has_wifi",
    "internet":         "has_wifi",
    "gym":              "has_gym",
    "fitness":          "has_gym",
    "study room":       "has_study_room",
    "study area":       "has_study_room",
    "cafeteria":        "has_cafeteria",
    "canteen":          "has_cafeteria",
    "laundry":          "has_laundry",
    "washing":          "has_laundry",
    "ac":               "has_ac",
    "air conditioning": "has_ac",
    "air conditioner":  "has_ac",
    "hot water":        "has_hot_water",
    "geyser":           "has_hot_water",
    "generator":        "has_generator",
    "backup":           "has_generator",
    "parking":          "has_parking",
    "prayer room":      "has_prayer_room",
    "mosque":           "has_prayer_room",
    "library":          "has_library",
    "cctv":             "has_cctv",
    "camera":           "has_cctv",
    "security guard":   "has_security_guard",
    "guard":            "has_security_guard",
    "common room":      "has_common_room",
}

AMENITY_LABELS = {
    "has_wifi":           "WiFi",
    "has_gym":            "Gym",
    "has_study_room":     "Study Room",
    "has_cafeteria":      "Cafeteria",
    "has_laundry":        "Laundry",
    "has_ac":             "AC",
    "has_hot_water":      "Hot Water",
    "has_generator":      "Generator",
    "has_parking":        "Parking",
    "has_prayer_room":    "Prayer Room",
    "has_library":        "Library",
    "has_cctv":           "CCTV",
    "has_security_guard": "Security Guard",
    "has_common_room":    "Common Room",
}

URDU_NUMBERS = {
    "10k": 10000, "12k": 12000, "15k": 15000, "20k": 20000,
    "8k":  8000,  "5k":  5000,  "25k": 25000, "30k": 30000,
    "das hazar":      10000, "barah hazar":     12000,
    "pandarah hazar": 15000, "paanch hazar":    5000,
    "bees hazar":     20000,
}

MULTI_WORD_AMENITIES = [
    "study room", "study area", "hot water", "air conditioning",
    "air conditioner", "prayer room", "common room", "security guard",
]

INTENT_EMOJI = {
    "hostel_search":    "🔍",
    "amenity_inquiry":  "🏷️",
    "pricing_info":     "💰",
    "booking_process":  "📋",
    "location_info":    "📍",
    "complaint":        "⚠️",
    "general_info":     "ℹ️",
}


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def load_intent_model():
    """Load the fine-tuned DistilBERT intent classifier."""
    model_path = str(MODEL_DIR)
    tokenizer  = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
    model      = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    le = joblib.load(str(_LABEL_PATH))
    return tokenizer, model, le


def load_spacy():
    """Load spaCy for entity extraction."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


# ══════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def classify_intent(text, tokenizer, model, le):
    """
    Run DistilBERT inference on input text.
    Returns (intent_label, confidence, all_scores_dict).
    INTELLIGENT: uses softmax probabilities, not argmax alone.
    """
    encoding = tokenizer(
        text,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids      = encoding["input_ids"],
            attention_mask = encoding["attention_mask"]
        ).logits

    probs      = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    intent     = le.classes_[pred_idx]

    all_scores = {
        le.classes_[i]: float(probs[i])
        for i in range(len(le.classes_))
    }
    return intent, confidence, all_scores


# ══════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_entities(text, nlp, hostels_df):
    """
    Extract structured entities from raw text using:
      - spaCy NER (budget via MONEY label, location via GPE/LOC)
      - regex patterns (budget, room type, distance)
      - keyword matching (amenities, hostel names)
    """
    entities   = {}
    text_lower = text.lower()

    # ── spaCy NER ─────────────────────────────────────────────────
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                entities["location_ref"] = ent.text
            if ent.label_ == "MONEY":
                amount = re.sub(r"[^\d]", "", ent.text)
                if amount and int(amount) > 1000:
                    entities["budget"] = int(amount)

    # ── Budget regex fallback ─────────────────────────────────────
    if "budget" not in entities:
        # Matches: "under 15000", "Rs. 8,000", "20000 rupees"
        match = re.search(
            r'(?:under|below|less than|upto|up to|within|max|maximum)?\s*'
            r'(?:rs\.?|pkr|rupees?)?\s*(\d[\d,]+)\s*(?:rs\.?|pkr|rupees?)?',
            text_lower
        )
        if match:
            val = int(match.group(1).replace(",", ""))
            if 1000 < val < 100000:
                entities["budget"] = val
        # Urdu shorthand
        for word, value in URDU_NUMBERS.items():
            if word in text_lower:
                entities["budget"] = value
                break

    # ── Gender detection ──────────────────────────────────────────
    if re.search(r'\b(girls?|female|women|ladies|larkiyon)\b', text_lower):
        entities["gender"] = "Female"
    elif re.search(r'\b(boys?|male|men|larkon|gents)\b', text_lower):
        entities["gender"] = "Male"

    # ── Room type ─────────────────────────────────────────────────
    room = re.search(r'\b(single|double|dorm|dormitory|shared)\b', text_lower)
    if room:
        rt = room.group(1)
        entities["room_type"] = "Dormitory" if rt in ("dorm","dormitory","shared") else rt.capitalize()

    # ── Distance ──────────────────────────────────────────────────
    dist = re.search(r'within\s*(\d+\.?\d*)\s*km', text_lower)
    if dist:
        entities["max_distance_km"] = float(dist.group(1))

    # ── Amenities ─────────────────────────────────────────────────
    found = []
    for kw in MULTI_WORD_AMENITIES:
        if kw in text_lower and AMENITY_MAP[kw] not in found:
            found.append(AMENITY_MAP[kw])
    for kw, col in AMENITY_MAP.items():
        if kw not in MULTI_WORD_AMENITIES:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                if col not in found:
                    found.append(col)
    if found:
        entities["amenities"] = list(set(found))

    # ── Hostel name ───────────────────────────────────────────────
    if hostels_df is not None:
        known = hostels_df["hostel_name"].str.lower().tolist()
        for name in known:
            if name in text_lower:
                entities["hostel_name"] = name
                break

    # ── Area / sector ─────────────────────────────────────────────
    areas = ["g-13","g-14","g13","g14","h-11","h11","f-10","f10",
             "e-11","e11","i-8","i8","bahria","dha","gulberg","pwd"]
    for area in areas:
        if area in text_lower:
            entities["area"] = area.upper().replace("G13","G-13")\
                .replace("G14","G-14").replace("H11","H-11")\
                .replace("F10","F-10").replace("E11","E-11")\
                .replace("I8","I-8")
            break

    return entities


# ══════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════

class ConversationContext:
    """
    Maintains state across turns.
    INTELLIGENT: bot remembers gender, budget, preferences
    so user doesn't have to repeat themselves.
    """
    def __init__(self):
        self.gender          = None
        self.budget          = None
        self.room_type       = None
        self.max_distance_km = None
        self.amenities       = []
        self.area            = None
        self.last_intent     = None
        self.last_hostels    = []   # last recommended hostel names
        self.turn_count      = 0

    def update(self, entities: dict):
        """Merge new entities into context, never overwrite with None."""
        if entities.get("gender"):
            self.gender = entities["gender"]
        if entities.get("budget"):
            self.budget = entities["budget"]
        if entities.get("room_type"):
            self.room_type = entities["room_type"]
        if entities.get("max_distance_km"):
            self.max_distance_km = entities["max_distance_km"]
        if entities.get("amenities"):
            # merge, don't replace
            for a in entities["amenities"]:
                if a not in self.amenities:
                    self.amenities.append(a)
        if entities.get("area"):
            self.area = entities["area"]

    def resolve(self, entities: dict) -> dict:
        """Fill gaps in current entities from context memory."""
        merged = dict(entities)
        if not merged.get("gender")          and self.gender:
            merged["gender"]          = self.gender
        if not merged.get("budget")          and self.budget:
            merged["budget"]          = self.budget
        if not merged.get("room_type")       and self.room_type:
            merged["room_type"]       = self.room_type
        if not merged.get("max_distance_km") and self.max_distance_km:
            merged["max_distance_km"] = self.max_distance_km
        if not merged.get("amenities")       and self.amenities:
            merged["amenities"]       = self.amenities
        return merged

    def summary(self) -> str:
        parts = []
        if self.gender:          parts.append(f"Gender: {self.gender}")
        if self.budget:          parts.append(f"Budget: PKR {self.budget:,}")
        if self.room_type:       parts.append(f"Room: {self.room_type}")
        if self.max_distance_km: parts.append(f"Distance: ≤{self.max_distance_km}km")
        if self.amenities:
            labels = [AMENITY_LABELS.get(a, a) for a in self.amenities]
            parts.append(f"Amenities: {', '.join(labels)}")
        return " · ".join(parts) if parts else "No preferences set yet"

    def clear(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════════
# RESPONSE GENERATORS — one per intent
# ══════════════════════════════════════════════════════════════════

def respond_hostel_search(entities, hostels_df, rec_fn=None):
    """
    INTELLIGENT: calls Eraj's hybrid recommendation engine if available,
    otherwise falls back to direct CSV filtering with scoring.
    """
    gender  = entities.get("gender", "Male")
    budget  = entities.get("budget")
    max_dist= entities.get("max_distance_km", 5.0)
    amenities = entities.get("amenities", [])
    room_type = entities.get("room_type", "Single")
    area    = entities.get("area")

    hostel_type = "Girls" if gender == "Female" else "Boys"
    df = hostels_df[hostels_df["hostel_type"] == hostel_type].copy()

    used_engine = False

    # ── Try hybrid engine first ───────────────────────────────────
    if rec_fn is not None:
        try:
            must_have = [AMENITY_LABELS.get(a, a) for a in amenities]
            recs = rec_fn(
                gender      = gender,
                department  = "Computer Science",
                budget_max  = budget or 25000,
                max_dist    = max_dist,
                study_pref  = 0.6,
                food_pref   = "Both",
                room_type   = room_type,
                price_sens  = 0.6,
                comfort_pref= 0.5,
                noise_tol   = 0.3,
                curfew_flex = 0.5,
                needs_transport = (max_dist > 3.0),
                must_have   = must_have,
                top_k       = MAX_RESULTS,
            )
            used_engine = True
        except Exception:
            recs = None
    else:
        recs = None

    # ── Fallback: direct CSV filter + simple scoring ──────────────
    if recs is None or (hasattr(recs, '__len__') and len(recs) == 0):
        used_engine = False
        if budget:
            df = df[df["single_room_price"] <= budget]
        df = df[df["distance_from_fast_km"] <= max_dist]
        for amenity in amenities:
            if amenity in df.columns:
                df = df[df[amenity] == 1]
        if area:
            area_clean = area.replace("-","").lower()
            df = df[df["area"].str.replace("-","").str.lower().str.contains(area_clean, na=False)]
        df["_score"] = df["overall_rating"] / 5.0
        if budget:
            df["_score"] += (1 - df["single_room_price"] / budget).clip(0,1) * 0.3
        df["_score"] += (1 - df["distance_from_fast_km"] / 6.14) * 0.2
        df = df.sort_values("_score", ascending=False).head(MAX_RESULTS)
        recs = df

    if recs is None or len(recs) == 0:
        return {
            "type":    "no_results",
            "message": (
                f"😔 No {hostel_type.lower()} hostels found matching your criteria.\n\n"
                f"Try relaxing your filters — increase the budget, "
                f"expand the distance, or remove some amenity requirements."
            ),
            "hostels": [],
            "used_engine": used_engine,
        }

    hostel_names = recs["hostel_name"].tolist()
    cards = []
    for _, row in recs.iterrows():
        card = {
            "name":     row["hostel_name"],
            "type":     row["hostel_type"],
            "area":     row["area"],
            "price":    int(row["single_room_price"]),
            "rating":   row["overall_rating"],
            "distance": row["distance_from_fast_km"],
            "security": row["security_rating"],
        }
        if used_engine and "hybrid_score" in row:
            card["score"] = round(row["hybrid_score"] * 100, 1)
        amenity_list = []
        for col, lbl in AMENITY_LABELS.items():
            if col in row and row[col] == 1:
                amenity_list.append(lbl)
        card["amenities"] = amenity_list[:6]
        cards.append(card)

    summary_parts = [f"{hostel_type} hostels"]
    if budget:    summary_parts.append(f"≤ PKR {budget:,}")
    if max_dist != 5.0: summary_parts.append(f"≤ {max_dist}km")
    if amenities: summary_parts.append(", ".join(AMENITY_LABELS.get(a,a) for a in amenities))

    return {
        "type":        "hostel_results",
        "message":     f"Found **{len(cards)}** {' · '.join(summary_parts)}",
        "hostels":     cards,
        "used_engine": used_engine,
        "names":       hostel_names,
    }


def respond_amenity_inquiry(entities, hostels_df):
    """Query hostels.csv for amenity availability."""
    amenities    = entities.get("amenities", [])
    hostel_name  = entities.get("hostel_name")
    gender       = entities.get("gender")

    # ── Specific hostel ───────────────────────────────────────────
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) == 0:
            return {"type": "text", "message": f"❓ Couldn't find a hostel named '{hostel_name}'."}
        row = row.iloc[0]
        if amenities:
            lines = []
            for a in amenities:
                lbl  = AMENITY_LABELS.get(a, a)
                have = bool(row.get(a, 0))
                lines.append(f"{'✅' if have else '❌'} **{lbl}**: {'Available' if have else 'Not available'}")
            msg = f"**{row['hostel_name']}** amenities:\n\n" + "\n".join(lines)
        else:
            have = [AMENITY_LABELS[c] for c in AMENITY_LABELS if row.get(c, 0) == 1]
            msg  = f"**{row['hostel_name']}** has: {', '.join(have) if have else 'No standard amenities listed'}"
        return {"type": "text", "message": msg}

    # ── Which hostels have this amenity ───────────────────────────
    if amenities:
        df = hostels_df.copy()
        if gender:
            df = df[df["hostel_type"] == ("Girls" if gender=="Female" else "Boys")]
        for a in amenities:
            if a in df.columns:
                df = df[df[a] == 1]
        df = df.sort_values("overall_rating", ascending=False)
        lbl_list = [AMENITY_LABELS.get(a,a) for a in amenities]
        if len(df) == 0:
            return {"type": "text", "message": f"😔 No hostels found with {' + '.join(lbl_list)}."}
        names = df["hostel_name"].head(5).tolist()
        msg = (
            f"**{len(df)} hostels** have {' + '.join(lbl_list)}:\n\n"
            + "\n".join(f"• {n} ({df[df['hostel_name']==n]['area'].values[0]}, "
                        f"⭐ {df[df['hostel_name']==n]['overall_rating'].values[0]})"
                        for n in names)
        )
        if len(df) > 5:
            msg += f"\n\n_...and {len(df)-5} more. Use Find My Hostel for full results._"
        return {"type": "text", "message": msg}

    return {"type": "text", "message": "Which amenity are you asking about? (WiFi, gym, study room, AC, generator, etc.)"}


def respond_pricing_info(entities, hostels_df):
    """Return fee information from hostels.csv."""
    hostel_name = entities.get("hostel_name")
    budget      = entities.get("budget")
    gender      = entities.get("gender")
    room_type   = entities.get("room_type", "single").lower()

    # ── Specific hostel ───────────────────────────────────────────
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) == 0:
            return {"type": "text", "message": f"❓ Couldn't find '{hostel_name}'."}
        row   = row.iloc[0]
        lines = [f"**{row['hostel_name']}** — Fee Breakdown:"]
        lines.append(f"• Single room: **PKR {int(row['single_room_price']):,}/mo**")
        if pd.notna(row.get("double_room_price")) and row["double_room_price"] > 0:
            lines.append(f"• Double room: **PKR {int(row['double_room_price']):,}/mo**")
        if pd.notna(row.get("dorm_room_price")) and row["dorm_room_price"] > 0:
            lines.append(f"• Dorm bed: **PKR {int(row['dorm_room_price']):,}/mo**")
        lines.append(f"• Electricity: **{'Included ✅' if row['electricity_included'] else 'Separate (~PKR 2,000)'}**")
        lines.append(f"• Meals: **{'Included ✅' if row['meal_included'] else 'Not included'}**")
        elec  = 0 if row["electricity_included"] else 2000
        meals = 0 if not row["meal_included"] else 4000
        total = int(row["single_room_price"]) + elec + meals
        lines.append(f"\n💰 **Estimated total: PKR {total:,}/mo**")
        return {"type": "text", "message": "\n".join(lines)}

    # ── Cheapest options ──────────────────────────────────────────
    df = hostels_df.copy()
    if gender:
        df = df[df["hostel_type"] == ("Girls" if gender=="Female" else "Boys")]
    if budget:
        df = df[df["single_room_price"] <= budget]

    price_col = {
        "double":    "double_room_price",
        "dorm":      "dorm_room_price",
        "dormitory": "dorm_room_price",
    }.get(room_type, "single_room_price")

    df = df.sort_values("single_room_price").head(5)
    if len(df) == 0:
        return {"type":"text","message":f"😔 No hostels found under PKR {budget:,}."}

    price_range = f"PKR {int(df['single_room_price'].min()):,} – {int(df['single_room_price'].max()):,}"
    lines = [f"**Cheapest options** ({price_range}/mo):\n"]
    for _, row in df.iterrows():
        elec  = "⚡ incl." if row["electricity_included"] else ""
        meals = "🍽️ meals incl." if row["meal_included"] else ""
        extras = "  ".join(filter(None, [elec, meals]))
        lines.append(
            f"• **{row['hostel_name']}** — PKR {int(row['single_room_price']):,}/mo  "
            f"⭐{row['overall_rating']}  {extras}"
        )
    return {"type": "text", "message": "\n".join(lines)}


def respond_location_info(entities, hostels_df):
    """Return distance and location data from hostels.csv."""
    hostel_name = entities.get("hostel_name")
    max_dist    = entities.get("max_distance_km", 2.0)
    area        = entities.get("area")
    gender      = entities.get("gender")

    # ── Specific hostel distance ──────────────────────────────────
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) == 0:
            return {"type":"text","message":f"❓ Couldn't find '{hostel_name}'."}
        row = row.iloc[0]
        transport = "Yes ✅" if row["transport_nearby"] else "No ❌"
        msg = (
            f"**{row['hostel_name']}** location:\n\n"
            f"📍 Area: **{row['area']}**, Islamabad\n"
            f"📏 Distance from FAST H-11: **{row['distance_from_fast_km']} km**\n"
            f"🚌 Transport nearby: **{transport}**\n"
            f"🗺️ GPS: `{row['latitude']:.4f}, {row['longitude']:.4f}`"
        )
        walk_min = round(row["distance_from_fast_km"] / 5.0 * 60)
        msg += f"\n🚶 Walking: ~**{walk_min} min**"
        return {"type":"text","message": msg}

    # ── Filter by distance ────────────────────────────────────────
    df = hostels_df.copy()
    if gender:
        df = df[df["hostel_type"] == ("Girls" if gender=="Female" else "Boys")]
    if area:
        df = df[df["area"].str.lower().str.contains(area.lower(), na=False)]
    df = df[df["distance_from_fast_km"] <= max_dist].sort_values("distance_from_fast_km")

    if len(df) == 0:
        return {"type":"text","message":f"😔 No hostels found within {max_dist}km. Try increasing the distance."}

    lines = [f"**{len(df)} hostels within {max_dist}km** of FAST H-11:\n"]
    for _, row in df.head(6).iterrows():
        lines.append(
            f"• **{row['hostel_name']}** — {row['distance_from_fast_km']}km  "
            f"({row['area']})  ⭐{row['overall_rating']}"
        )
    return {"type":"text","message":"\n".join(lines)}


# ══════════════════════════════════════════════════════════════════
# OLLAMA — LOCAL LLM FOR INTELLIGENT OPEN-ENDED RESPONSES
# ══════════════════════════════════════════════════════════════════

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"   # fast 3B model, change to "mistral" if preferred

def _build_system_prompt(hostels_df: pd.DataFrame) -> str:
    """
    Build a rich system prompt with live dataset facts so Ollama
    can answer factual questions (counts, stats, policies) accurately.
    """
    n_girls  = len(hostels_df[hostels_df["hostel_type"] == "Girls"])
    n_boys   = len(hostels_df[hostels_df["hostel_type"] == "Boys"])
    total    = len(hostels_df)
    min_p    = int(hostels_df["single_room_price"].min())
    max_p    = int(hostels_df["single_room_price"].max())
    avg_p    = int(hostels_df["single_room_price"].mean())
    top3     = hostels_df.nlargest(3, "overall_rating")[["hostel_name","overall_rating","hostel_type"]].to_dict("records")
    areas    = sorted(hostels_df["area"].unique().tolist())
    n_wifi   = int(hostels_df["has_wifi"].sum())
    n_gym    = int(hostels_df["has_gym"].sum())
    n_study  = int(hostels_df["has_study_room"].sum())
    n_gen    = int(hostels_df["has_generator"].sum())
    n_ac     = int(hostels_df["has_ac"].sum())
    closest  = hostels_df.nsmallest(3,"distance_from_fast_km")[["hostel_name","distance_from_fast_km","hostel_type"]].to_dict("records")

    top3_str    = ", ".join(f"{r['hostel_name']} ({r['overall_rating']}/5, {r['hostel_type']})" for r in top3)
    closest_str = ", ".join(f"{r['hostel_name']} ({r['distance_from_fast_km']}km, {r['hostel_type']})" for r in closest)

    return f"""You are StayBuddy Assistant, an intelligent chatbot for FAST NUCES Islamabad students looking for hostels.

LIVE DATASET FACTS (always use these, never make up numbers):
- Total hostels: {total} ({n_girls} girls, {n_boys} boys)
- Price range: PKR {min_p:,} – {max_p:,}/month, average PKR {avg_p:,}
- Areas covered: {', '.join(areas)}
- Top rated: {top3_str}
- Closest to FAST H-11: {closest_str}
- Amenity counts: WiFi={n_wifi}, Gym={n_gym}, Study Room={n_study}, Generator={n_gen}, AC={n_ac}

BOOKING PROCESS:
1. Find a hostel using StayBuddy, note warden contact
2. Call/WhatsApp warden to confirm availability
3. Visit before paying any advance
4. Documents: CNIC copy, student ID, parent CNIC, 2 passport photos
5. Typical advance: 1-2 months rent as security deposit
6. Always get a receipt for payments

COMPLAINT PROCESS:
1. Talk to warden directly first (most issues resolve here)
2. If unresolved in 48 hours, escalate to hostel owner/management
3. Document everything with photos and dates
4. For safety issues (theft, harassment): contact FAST student affairs immediately
5. For maintenance: request a written ticket

HOSTEL RULES (general):
- Curfew typically 9pm-midnight (varies per hostel)
- Male visitors generally not allowed in girls hostels
- Cooking restrictions vary by hostel
- Noise quiet hours usually 10pm onwards
- CNIC registration required at check-in

Respond in 2-4 sentences max. Be helpful, friendly, and specific. Use the live data facts above when answering count or stats questions. Do NOT make up hostel names or prices."""


def call_ollama(user_text: str, intent: str, entities: dict,
                hostels_df: pd.DataFrame, context: "ConversationContext") -> str:
    """
    Send message to local Ollama instance.
    Enriches the prompt with intent, entities and context so the LLM
    gives a focused, data-grounded answer.
    """
    import urllib.request, json as _json, urllib.error

    system_prompt = _build_system_prompt(hostels_df)

    # Build context string for the LLM
    ctx_parts = []
    if context.gender:    ctx_parts.append(f"student gender: {context.gender}")
    if context.budget:    ctx_parts.append(f"budget: PKR {context.budget:,}")
    if entities.get("hostel_name"):
        hn  = entities["hostel_name"]
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hn]
        if len(row) > 0:
            r = row.iloc[0]
            ctx_parts.append(
                f"hostel context: {r['hostel_name']} in {r['area']}, "
                f"PKR {int(r['single_room_price']):,}/mo, "
                f"warden: {r['warden_contact_phone']}, "
                f"responsiveness: {r['warden_responsiveness']}/5"
            )

    ctx_str = (" | ".join(ctx_parts) + "\n") if ctx_parts else ""
    full_prompt = f"{ctx_str}User question (intent={intent}): {user_text}"

    payload = _json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": full_prompt,
        "system": system_prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 200},
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data    = payload,
        headers = {"Content-Type": "application/json"},
        method  = "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            result = _json.loads(resp.read().decode())
            return result.get("response", "").strip()
    except urllib.error.URLError:
        return None   # Ollama not running → caller uses fallback


def _ollama_available() -> bool:
    """Quick ping to check if Ollama is running locally."""
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return True
    except Exception:
        return False


# ── Fallback responses (used only when Ollama is offline) ─────────

def _fallback_booking(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    msg = ("**How to Book:** Call/WhatsApp the warden → visit before paying → "
           "bring CNIC copy, student ID, parent CNIC, 2 photos → "
           "pay 1–2 months advance, always get a receipt.")
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) > 0:
            msg += f"\n\n📞 **{row.iloc[0]['hostel_name']} Warden:** `{row.iloc[0]['warden_contact_phone']}`"
    return msg

def _fallback_complaint(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    msg = ("**Complaint Steps:** Talk to warden first → if unresolved in 48h escalate to owner → "
           "document with photos → for safety issues contact FAST student affairs immediately.")
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) > 0:
            msg += (f"\n\n📞 **{row.iloc[0]['hostel_name']} Warden:** `{row.iloc[0]['warden_contact_phone']}`"
                    f"\n⭐ Responsiveness: **{row.iloc[0]['warden_responsiveness']}/5**")
    return msg

def _fallback_general(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) > 0:
            r = row.iloc[0]
            curfew = f"{int(r['curfew_hour']):02d}:00" if r["curfew_hour"] > 0 else "No curfew"
            return (f"**{r['hostel_name']}** — ⭐{r['overall_rating']} · 🔒{r['security_rating']} · "
                    f"🧹{r['cleanliness_rating']} · Curfew: {curfew} · "
                    f"Capacity: {r['capacity']} · 📞`{r['warden_contact_phone']}`")
    n_g = len(hostels_df[hostels_df["hostel_type"]=="Girls"])
    n_b = len(hostels_df[hostels_df["hostel_type"]=="Boys"])
    return (f"StayBuddy covers **{len(hostels_df)} hostels** ({n_g} girls, {n_b} boys) "
            f"near FAST NUCES H-11, Islamabad. Price range PKR 8,152–39,833/month.")


def respond_with_llm(intent: str, user_text: str, entities: dict,
                     hostels_df: pd.DataFrame, context: "ConversationContext") -> dict:
    """
    Route booking_process / complaint / general_info / low_confidence
    through Ollama. Falls back gracefully if Ollama is offline.
    """
    ollama_reply = None
    if _ollama_available():
        ollama_reply = call_ollama(user_text, intent, entities, hostels_df, context)

    if ollama_reply:
        return {
            "type":    "text",
            "message": ollama_reply,
            "source":  "ollama",
        }

    # ── Ollama offline → structured fallback ─────────────────────
    if intent == "booking_process":
        msg = _fallback_booking(entities, hostels_df)
    elif intent == "complaint":
        msg = _fallback_complaint(entities, hostels_df)
    elif intent == "general_info":
        msg = _fallback_general(entities, hostels_df)
    else:
        # low confidence fallback
        msg = ("I'm not sure I understood that fully. Try: "
               "_\"Show me girls hostels under 15k\"_, "
               "_\"Does Khadija Residence have WiFi?\"_, or "
               "_\"How do I book a hostel?\"_")

    return {
        "type":    "text",
        "message": msg,
        "source":  "fallback",
    }


# ══════════════════════════════════════════════════════════════════
# MAIN CHAT FUNCTION — called by Streamlit
# ══════════════════════════════════════════════════════════════════

def chat(
    user_text: str,
    context: ConversationContext,
    tokenizer, model, le,
    nlp,
    hostels_df: pd.DataFrame,
    rec_fn=None,
) -> dict:
    """
    Full pipeline:
      text → intent (DistilBERT) → entities (spaCy+regex)
      → context merge → route to intent handler → response dict

    Architecture:
      hostel_search          → Eraj's hybrid engine   ✅ intelligent
      amenity/pricing/location → CSV query             ✅ intelligent
      booking/complaint/general → Ollama local LLM    ✅ intelligent
      low confidence           → Ollama LLM fallback  ✅ intelligent
    """
    context.turn_count += 1

    # 1. Classify intent
    intent, confidence, all_scores = classify_intent(user_text, tokenizer, model, le)
    context.last_intent = intent

    # 2. Extract entities
    raw_entities = extract_entities(user_text, nlp, hostels_df)

    # 3. Update + resolve context
    context.update(raw_entities)
    entities = context.resolve(raw_entities)

    # 4. Route
    if confidence < CONFIDENCE_THRESHOLD:
        # Low confidence → Ollama handles it intelligently
        response = respond_with_llm("unknown", user_text, entities, hostels_df, context)
        response["type"] = "clarification"

    elif intent == "hostel_search":
        response = respond_hostel_search(entities, hostels_df, rec_fn)

    elif intent == "amenity_inquiry":
        response = respond_amenity_inquiry(entities, hostels_df)

    elif intent == "pricing_info":
        response = respond_pricing_info(entities, hostels_df)

    elif intent == "location_info":
        response = respond_location_info(entities, hostels_df)

    elif intent in ("booking_process", "complaint", "general_info"):
        # Ollama LLM — truly intelligent, data-grounded, not hardcoded
        response = respond_with_llm(intent, user_text, entities, hostels_df, context)

    else:
        response = respond_with_llm("unknown", user_text, entities, hostels_df, context)

    # 5. Track last recommended hostels for context
    if response.get("names"):
        context.last_hostels = response["names"]

    # 6. Attach metadata for Streamlit display
    response["intent"]     = intent
    response["confidence"] = confidence
    response["all_scores"] = all_scores
    response["entities"]   = raw_entities
    response["emoji"]      = INTENT_EMOJI.get(intent, "💬")

    return response
