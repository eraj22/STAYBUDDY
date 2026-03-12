"""
╔══════════════════════════════════════════════════════════════════╗
║   StayBuddy — NLP Chatbot Core  v2.0                            ║
║   Author  : Samiya Saleem (22I-1065)                             ║
║   Model   : DistilBERT fine-tuned · 7 intents · 84.75% acc      ║
╚══════════════════════════════════════════════════════════════════╝

INTELLIGENT features:
  1. DistilBERT transformer (not keyword matching)
  2. Confidence threshold — asks clarification if unsure
  3. spaCy + regex entity extraction
  4. Multi-turn context — remembers ALL preferences across turns
  5. Follow-up resolution — "which of these", "how many of them",
     "show me the cheapest one" all resolve against previous results
  6. hostel_search -> Eraj's hybrid recommendation engine
  7. Stats queries — "how many girls hostels are there"
  8. Comparative queries — "which is closest / cheapest / highest rated"
"""

import re
import os
import json
import joblib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import spacy

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.resolve()
MODEL_DIR = BASE_DIR / "intent_model"
DATA_DIR  = BASE_DIR / "data"

_CLEAN_BASE = Path("C:/staybuddy_models")
if " " in str(BASE_DIR) and _CLEAN_BASE.exists():
    MODEL_DIR   = _CLEAN_BASE / "intent_model"
    _LABEL_PATH = _CLEAN_BASE / "label_encoder.pkl"
else:
    _LABEL_PATH = BASE_DIR / "label_encoder.pkl"

# ── Constants ──────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55
MAX_RESULTS          = 5

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
    "transport":        "transport_nearby",
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
    "transport_nearby":   "Transport",
}

URDU_NUMBERS = {
    "10k": 10000, "12k": 12000, "15k": 15000, "20k": 20000,
    "8k":  8000,  "5k":  5000,  "25k": 25000, "30k": 30000,
    "das hazar":      10000, "barah hazar":  12000,
    "pandarah hazar": 15000, "paanch hazar": 5000,
    "bees hazar":     20000,
}

MULTI_WORD_AMENITIES = [
    "study room", "study area", "hot water", "air conditioning",
    "air conditioner", "prayer room", "common room", "security guard",
]

INTENT_EMOJI = {
    "hostel_search":   "🔍",
    "amenity_inquiry": "🏷️",
    "pricing_info":    "💰",
    "booking_process": "📋",
    "location_info":   "📍",
    "complaint":       "⚠️",
    "general_info":    "ℹ️",
    "stats_query":     "📊",
    "followup":        "↩️",
}

FOLLOWUP_PATTERNS = [
    r'\b(which|what|which one|which ones)\b.*(of these|of them|from these|from them|among them|among these)',
    r'\b(how many|kitne|kitni).*(of these|of them|from these|from them)',
    r'\b(show me|give me|tell me).*(cheapest|closest|nearest|best|highest|lowest|top).*(one|ones|of these|of them)',
    r'\b(the cheapest|the closest|the nearest|the best|the highest rated|the top rated)\b',
    r'\b(filter|narrow|refine|from the list|from those|from above)\b',
    r'\b(do any of them|does any of them|do they|any of these)\b',
    r'\bsabse (sasta|mehnga|kareeb|acha)\b',
]

STATS_PATTERNS = [
    r'\bhow many\b',
    r'\bkitne\b',
    r'\bkitni\b',
    r'\btotal (number|count|hostels)\b',
    r'\bcount\b',
]


# ══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def load_intent_model():
    model_path = str(MODEL_DIR)
    tokenizer  = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
    model      = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
    le = joblib.load(str(_LABEL_PATH))
    return tokenizer, model, le


def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


# ══════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════

def classify_intent(text, tokenizer, model, le):
    encoding = tokenizer(
        text, max_length=64, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        ).logits
    probs      = torch.softmax(logits, dim=1).squeeze().numpy()
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    intent     = le.classes_[pred_idx]
    all_scores = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}
    return intent, confidence, all_scores


# ══════════════════════════════════════════════════════════════════
# FOLLOW-UP & STATS DETECTION
# ══════════════════════════════════════════════════════════════════

def is_followup(text: str) -> bool:
    t = text.lower()
    for pat in FOLLOWUP_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def is_stats_query(text: str) -> bool:
    t = text.lower()
    for pat in STATS_PATTERNS:
        if re.search(pat, t):
            return True
    return False


def detect_comparison(text: str):
    t = text.lower()
    if re.search(r'\b(cheapest|sasta|lowest price|least expensive|affordable)\b', t):
        return "cheapest"
    if re.search(r'\b(closest|nearest|kareeb|most nearby|shortest distance)\b', t):
        return "closest"
    if re.search(r'\b(best|highest rated|top rated|best rating)\b', t):
        return "best_rated"
    if re.search(r'\b(most expensive|mehnga|priciest)\b', t):
        return "most_expensive"
    if re.search(r'\b(safest|most secure|highest security)\b', t):
        return "safest"
    return None


# ══════════════════════════════════════════════════════════════════
# ENTITY EXTRACTION
# ══════════════════════════════════════════════════════════════════

def extract_entities(text: str, nlp, hostels_df) -> dict:
    entities   = {}
    text_lower = text.lower()

    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC", "FAC"]:
                entities["location_ref"] = ent.text
            if ent.label_ == "MONEY":
                amount = re.sub(r"[^\d]", "", ent.text)
                if amount and int(amount) > 1000:
                    entities["budget"] = int(amount)

    if "budget" not in entities:
        match = re.search(
            r'(?:under|below|less than|upto|up to|within|max|maximum)?\s*'
            r'(?:rs\.?|pkr|rupees?)?\s*(\d[\d,]+)\s*(?:rs\.?|pkr|rupees?)?',
            text_lower
        )
        if match:
            val = int(match.group(1).replace(",", ""))
            if 1000 < val < 100000:
                entities["budget"] = val
        for word, value in URDU_NUMBERS.items():
            if word in text_lower:
                entities["budget"] = value
                break

    if re.search(r'\b(girls?|female|women|ladies|larkiyon)\b', text_lower):
        entities["gender"] = "Female"
    elif re.search(r'\b(boys?|male|men|larkon|gents)\b', text_lower):
        entities["gender"] = "Male"

    room = re.search(r'\b(single|double|dorm|dormitory|shared)\b', text_lower)
    if room:
        rt = room.group(1)
        entities["room_type"] = "Dormitory" if rt in ("dorm","dormitory","shared") else rt.capitalize()

    dist = re.search(r'within\s*(\d+\.?\d*)\s*km', text_lower)
    if dist:
        entities["max_distance_km"] = float(dist.group(1))

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

    if hostels_df is not None:
        known = hostels_df["hostel_name"].str.lower().tolist()
        for name in known:
            if name in text_lower:
                entities["hostel_name"] = name
                break

    areas = ["g-13","g-14","g13","g14","h-11","h11","f-10","f10",
             "e-11","e11","i-8","i8","bahria","dha","gulberg","pwd","i-10","i10"]
    for area in areas:
        if area in text_lower:
            entities["area"] = area.upper().replace("G13","G-13").replace("G14","G-14")\
                .replace("H11","H-11").replace("F10","F-10").replace("E11","E-11")\
                .replace("I8","I-8").replace("I10","I-10")
            break

    return entities


# ══════════════════════════════════════════════════════════════════
# CONTEXT MANAGER
# ══════════════════════════════════════════════════════════════════

class ConversationContext:
    def __init__(self):
        self.gender           = None
        self.budget           = None
        self.room_type        = None
        self.max_distance_km  = None
        self.amenities        = []
        self.area             = None
        self.last_intent      = None
        self.last_hostels     = []
        self.last_hostel_names= []
        self.turn_count       = 0
        self.started_at       = datetime.now()

    def update(self, entities: dict):
        if entities.get("gender"):          self.gender = entities["gender"]
        if entities.get("budget"):          self.budget = entities["budget"]
        if entities.get("room_type"):       self.room_type = entities["room_type"]
        if entities.get("max_distance_km"): self.max_distance_km = entities["max_distance_km"]
        if entities.get("amenities"):
            for a in entities["amenities"]:
                if a not in self.amenities:
                    self.amenities.append(a)
        if entities.get("area"):            self.area = entities["area"]

    def resolve(self, entities: dict) -> dict:
        merged = dict(entities)
        if not merged.get("gender")          and self.gender:          merged["gender"]          = self.gender
        if not merged.get("budget")          and self.budget:          merged["budget"]          = self.budget
        if not merged.get("room_type")       and self.room_type:       merged["room_type"]       = self.room_type
        if not merged.get("max_distance_km") and self.max_distance_km: merged["max_distance_km"] = self.max_distance_km
        if not merged.get("amenities")       and self.amenities:       merged["amenities"]       = self.amenities
        return merged

    def set_last_hostels(self, cards: list):
        self.last_hostels      = cards
        self.last_hostel_names = [c["name"] for c in cards]

    def get_last_hostel_df(self, hostels_df: pd.DataFrame) -> pd.DataFrame:
        if not self.last_hostel_names:
            return pd.DataFrame()
        return hostels_df[hostels_df["hostel_name"].isin(self.last_hostel_names)].copy()

    def summary(self) -> str:
        parts = []
        if self.gender:           parts.append(f"Gender: {self.gender}")
        if self.budget:           parts.append(f"Budget: PKR {self.budget:,}")
        if self.room_type:        parts.append(f"Room: {self.room_type}")
        if self.max_distance_km:  parts.append(f"Distance: ≤{self.max_distance_km}km")
        if self.amenities:
            labels = [AMENITY_LABELS.get(a, a) for a in self.amenities]
            parts.append(f"Amenities: {', '.join(labels)}")
        if self.last_hostel_names: parts.append(f"Last shown: {len(self.last_hostel_names)} hostels")
        return " · ".join(parts) if parts else "No preferences set yet"

    def clear(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════════
# FOLLOW-UP HANDLER
# ══════════════════════════════════════════════════════════════════

def respond_followup(user_text: str, context: ConversationContext,
                     hostels_df: pd.DataFrame, entities: dict) -> dict:
    t = user_text.lower()
    prev_df = context.get_last_hostel_df(hostels_df)

    if prev_df.empty:
        return {"type":"text","message":"❓ No previous results to filter. Search for hostels first."}

    n = len(prev_df)
    prev_names_str = ", ".join(context.last_hostel_names)

    # How many of them [have X / are within Y km]
    if re.search(r'\bhow many\b|\bkitne\b|\bkitni\b', t):
        amenities  = entities.get("amenities", [])
        dist_match = re.search(r'within\s*(\d+\.?\d*)\s*km', t)

        if amenities:
            filtered = prev_df.copy()
            for a in amenities:
                if a in filtered.columns:
                    filtered = filtered[filtered[a] == 1]
            lbl   = ", ".join(AMENITY_LABELS.get(a,a) for a in amenities)
            count = len(filtered)
            msg   = f"**{count} of the {n} hostels** have {lbl}."
            if count > 0:
                msg += "\n\n" + "\n".join(f"• {r['hostel_name']}" for _, r in filtered.iterrows())
            return {"type":"text","message": msg}

        if dist_match:
            km       = float(dist_match.group(1))
            filtered = prev_df[prev_df["distance_from_fast_km"] <= km]
            msg      = f"**{len(filtered)} of the {n} hostels** are within {km}km of FAST."
            if len(filtered) > 0:
                msg += "\n\n" + "\n".join(
                    f"• {r['hostel_name']} — {r['distance_from_fast_km']}km"
                    for _, r in filtered.iterrows()
                )
            return {"type":"text","message": msg}

        return {"type":"text","message":f"There are **{n} hostels** in the previous results."}

    # Which of these have [amenity]
    amenities = entities.get("amenities", [])
    if amenities:
        filtered = prev_df.copy()
        for a in amenities:
            if a in filtered.columns:
                filtered = filtered[filtered[a] == 1]
            elif a == "transport_nearby" and "transport_nearby" in filtered.columns:
                filtered = filtered[filtered["transport_nearby"] == 1]
        lbl = ", ".join(AMENITY_LABELS.get(a,a) for a in amenities)

        if len(filtered) == 0:
            return {"type":"text","message":f"❌ None of the {n} hostels have {lbl}."}

        cards = _build_cards(filtered)
        return {
            "type":        "hostel_results",
            "message":     f"**{len(filtered)} of the {n} hostels** have {lbl}:",
            "hostels":     cards,
            "names":       filtered["hostel_name"].tolist(),
            "used_engine": False,
        }

    # Superlative comparisons
    comparison = detect_comparison(user_text)
    if comparison:
        if comparison == "cheapest":
            row = prev_df.nsmallest(1, "single_room_price").iloc[0]
            msg = (f"💰 **Cheapest** from your results:\n\n"
                   f"**{row['hostel_name']}** — PKR {int(row['single_room_price']):,}/mo  "
                   f"⭐{row['overall_rating']}  📏{row['distance_from_fast_km']}km")
        elif comparison == "closest":
            row = prev_df.nsmallest(1, "distance_from_fast_km").iloc[0]
            msg = (f"📍 **Closest to FAST** from your results:\n\n"
                   f"**{row['hostel_name']}** — {row['distance_from_fast_km']}km  "
                   f"PKR {int(row['single_room_price']):,}/mo  ⭐{row['overall_rating']}")
        elif comparison == "best_rated":
            row = prev_df.nlargest(1, "overall_rating").iloc[0]
            msg = (f"⭐ **Highest rated** from your results:\n\n"
                   f"**{row['hostel_name']}** — ⭐{row['overall_rating']}/5  "
                   f"PKR {int(row['single_room_price']):,}/mo  📏{row['distance_from_fast_km']}km")
        elif comparison == "most_expensive":
            row = prev_df.nlargest(1, "single_room_price").iloc[0]
            msg = (f"💸 **Most expensive** from your results:\n\n"
                   f"**{row['hostel_name']}** — PKR {int(row['single_room_price']):,}/mo  "
                   f"⭐{row['overall_rating']}")
        elif comparison == "safest":
            row = prev_df.nlargest(1, "security_rating").iloc[0]
            msg = (f"🔒 **Most secure** from your results:\n\n"
                   f"**{row['hostel_name']}** — Security {row['security_rating']}/5  "
                   f"PKR {int(row['single_room_price']):,}/mo  ⭐{row['overall_rating']}")
        return {"type":"text","message": msg}

    return {
        "type": "text",
        "message": (
            f"Your previous results had **{n} hostels**: {prev_names_str}.\n\n"
            f"You can ask:\n"
            f'• _"Which of these have WiFi / AC / gym?"_\n'
            f'• _"How many of them are within 2km?"_\n'
            f'• _"Show me the cheapest / closest / highest rated"_'
        )
    }


# ══════════════════════════════════════════════════════════════════
# STATS HANDLER
# ══════════════════════════════════════════════════════════════════

def respond_stats(user_text: str, entities: dict, hostels_df: pd.DataFrame) -> dict:
    gender    = entities.get("gender")
    area      = entities.get("area")
    amenities = entities.get("amenities", [])

    df = hostels_df.copy()
    if gender:
        df = df[df["hostel_type"] == ("Girls" if gender=="Female" else "Boys")]
    if area:
        df = df[df["area"].str.lower().str.contains(area.lower(), na=False)]
    for a in amenities:
        if a in df.columns:
            df = df[df[a] == 1]

    count      = len(df)
    type_label = ("girls" if gender=="Female" else "boys") if gender else ""

    if gender and not amenities and not area:
        girls = len(hostels_df[hostels_df["hostel_type"]=="Girls"])
        boys  = len(hostels_df[hostels_df["hostel_type"]=="Boys"])
        msg   = (f"There are **{count} {type_label} hostels** in StayBuddy.\n\n"
                 f"📊 Full breakdown:\n"
                 f"• 🏠 Girls hostels: **{girls}**\n"
                 f"• 🏠 Boys hostels: **{boys}**\n"
                 f"• Total: **{girls+boys}**")
    elif amenities:
        lbl = ", ".join(AMENITY_LABELS.get(a,a) for a in amenities)
        msg = f"**{count} {type_label} hostels** have {lbl}."
        if 0 < count <= 8:
            names = df["hostel_name"].tolist()
            msg  += "\n\n" + "\n".join(f"• {n} ({df[df['hostel_name']==n]['area'].values[0]})" for n in names)
    elif area:
        msg = f"**{count} hostels** are in the {area} area."
    else:
        girls = len(hostels_df[hostels_df["hostel_type"]=="Girls"])
        boys  = len(hostels_df[hostels_df["hostel_type"]=="Boys"])
        msg   = (f"StayBuddy has **{len(hostels_df)} hostels** in total:\n\n"
                 f"• 🏠 Girls hostels: **{girls}**\n"
                 f"• 🏠 Boys hostels: **{boys}**\n"
                 f"• Price range: **PKR 8,152 – 39,833/mo**\n"
                 f"• Ratings: **3.1 – 4.5 ⭐**\n"
                 f"• Areas: G-13, H-11, F-10, E-11, I-8, I-10, Bahria Town, DHA, Gulberg and more")

    return {"type":"text","message": msg}


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _build_cards(df: pd.DataFrame) -> list:
    cards = []
    for _, row in df.iterrows():
        card = {
            "name":     row["hostel_name"],
            "type":     row["hostel_type"],
            "area":     row["area"],
            "price":    int(row["single_room_price"]),
            "rating":   row["overall_rating"],
            "distance": row["distance_from_fast_km"],
            "security": row["security_rating"],
        }
        amenity_list = [lbl for col, lbl in AMENITY_LABELS.items()
                        if col in row.index and row[col] == 1]
        card["amenities"] = amenity_list[:6]
        cards.append(card)
    return cards


# ══════════════════════════════════════════════════════════════════
# INTENT RESPONSE HANDLERS
# ══════════════════════════════════════════════════════════════════

def respond_hostel_search(entities, hostels_df, rec_fn=None):
    gender    = entities.get("gender", "Male")
    budget    = entities.get("budget")
    max_dist  = entities.get("max_distance_km", 5.0)
    amenities = entities.get("amenities", [])
    room_type = entities.get("room_type", "Single")
    area      = entities.get("area")

    hostel_type = "Girls" if gender == "Female" else "Boys"
    df          = hostels_df[hostels_df["hostel_type"] == hostel_type].copy()
    used_engine = False
    recs        = None

    if rec_fn is not None:
        try:
            must_have = [AMENITY_LABELS.get(a, a) for a in amenities]
            recs = rec_fn(
                gender=gender, department="Computer Science",
                budget_max=budget or 25000, max_dist=max_dist,
                study_pref=0.6, food_pref="Both", room_type=room_type,
                price_sens=0.6, comfort_pref=0.5, noise_tol=0.3,
                curfew_flex=0.5, needs_transport=(max_dist > 3.0),
                must_have=must_have, top_k=MAX_RESULTS,
            )
            used_engine = True
        except Exception:
            recs = None

    if recs is None or (hasattr(recs, '__len__') and len(recs) == 0):
        used_engine = False
        if budget:   df = df[df["single_room_price"] <= budget]
        df = df[df["distance_from_fast_km"] <= max_dist]
        for a in amenities:
            if a in df.columns:
                df = df[df[a] == 1]
        if area:
            df = df[df["area"].str.lower().str.contains(area.lower().replace("-",""), na=False)]
        df["_score"] = df["overall_rating"] / 5.0
        if budget:
            df["_score"] += (1 - df["single_room_price"] / budget).clip(0,1) * 0.3
        df["_score"] += (1 - df["distance_from_fast_km"] / 6.14) * 0.2
        recs = df.sort_values("_score", ascending=False).head(MAX_RESULTS)

    if recs is None or len(recs) == 0:
        return {
            "type":"no_results",
            "message":(f"😔 No {hostel_type.lower()} hostels match your criteria.\n\n"
                       "Try relaxing filters — increase budget, expand distance, or remove amenities."),
            "hostels":[],"used_engine":False,
        }

    cards = _build_cards(recs)
    if used_engine and "hybrid_score" in recs.columns:
        for i, (_, row) in enumerate(recs.iterrows()):
            if i < len(cards):
                cards[i]["score"] = round(float(row["hybrid_score"]) * 100, 1)

    summary_parts = [f"{hostel_type} hostels"]
    if budget:          summary_parts.append(f"≤ PKR {budget:,}")
    if max_dist != 5.0: summary_parts.append(f"≤ {max_dist}km")
    if amenities:       summary_parts.append(", ".join(AMENITY_LABELS.get(a,a) for a in amenities))

    return {
        "type":        "hostel_results",
        "message":     f"Found **{len(cards)}** {' · '.join(summary_parts)}",
        "hostels":     cards,
        "used_engine": used_engine,
        "names":       [c["name"] for c in cards],
    }


def respond_amenity_inquiry(entities, hostels_df):
    amenities   = entities.get("amenities", [])
    hostel_name = entities.get("hostel_name")
    gender      = entities.get("gender")

    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) == 0:
            return {"type":"text","message":f"❓ Couldn't find '{hostel_name}'."}
        row = row.iloc[0]
        if amenities:
            lines = [f"**{row['hostel_name']}** ({row['area']}) amenities:\n"]
            for a in amenities:
                lbl  = AMENITY_LABELS.get(a, a)
                have = bool(row.get(a, 0))
                lines.append(f"{'✅' if have else '❌'} **{lbl}**: {'Available' if have else 'Not available'}")
            return {"type":"text","message":"\n".join(lines)}
        have = [AMENITY_LABELS[c] for c in AMENITY_LABELS if c in row.index and row[c] == 1]
        return {"type":"text","message":f"**{row['hostel_name']}** has: {', '.join(have) if have else 'No amenities listed'}"}

    if amenities:
        df = hostels_df.copy()
        if gender:
            df = df[df["hostel_type"] == ("Girls" if gender=="Female" else "Boys")]
        for a in amenities:
            if a in df.columns:
                df = df[df[a] == 1]
        df    = df.sort_values("overall_rating", ascending=False)
        lbls  = [AMENITY_LABELS.get(a,a) for a in amenities]
        if len(df) == 0:
            return {"type":"text","message":f"😔 No hostels found with {' + '.join(lbls)}."}
        names = df["hostel_name"].head(6).tolist()
        msg   = (f"**{len(df)} hostels** have {' + '.join(lbls)}:\n\n"
                 + "\n".join(f"• {n} ({df[df['hostel_name']==n]['area'].values[0]}, ⭐{df[df['hostel_name']==n]['overall_rating'].values[0]})" for n in names))
        if len(df) > 6: msg += f"\n\n_...and {len(df)-6} more._"
        return {"type":"text","message": msg}

    return {"type":"text","message":"Which amenity are you asking about? (WiFi, gym, study room, AC, generator, etc.)"}


def respond_pricing_info(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    budget      = entities.get("budget")
    gender      = entities.get("gender")

    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) == 0:
            return {"type":"text","message":f"❓ Couldn't find '{hostel_name}'."}
        row   = row.iloc[0]
        elec  = 0 if row["electricity_included"] else 2000
        total = int(row["single_room_price"]) + elec
        lines = [
            f"**{row['hostel_name']}** — Fee Breakdown:",
            f"• Single room: **PKR {int(row['single_room_price']):,}/mo**",
        ]
        if pd.notna(row.get("double_room_price")) and row["double_room_price"] > 0:
            lines.append(f"• Double room: **PKR {int(row['double_room_price']):,}/mo**")
        if pd.notna(row.get("dorm_room_price")) and row["dorm_room_price"] > 0:
            lines.append(f"• Dorm bed: **PKR {int(row['dorm_room_price']):,}/mo**")
        lines += [
            f"• Electricity: **{'Included ✅' if row['electricity_included'] else 'Separate (~PKR 2,000)'}**",
            f"• Meals: **{'Included ✅' if row['meal_included'] else 'Not included'}**",
            f"\n💰 **Estimated total: PKR {total:,}/mo**",
        ]
        return {"type":"text","message":"\n".join(lines)}

    df = hostels_df.copy()
    if gender: df = df[df["hostel_type"] == ("Girls" if gender=="Female" else "Boys")]
    if budget: df = df[df["single_room_price"] <= budget]
    df = df.sort_values("single_room_price").head(5)
    if len(df) == 0:
        return {"type":"text","message":f"😔 No hostels found under PKR {budget:,}."}
    rng   = f"PKR {int(df['single_room_price'].min()):,} – {int(df['single_room_price'].max()):,}"
    lines = [f"**Cheapest options** ({rng}/mo):\n"]
    for _, row in df.iterrows():
        extras = ("  ⚡ incl." if row["electricity_included"] else "") + \
                 ("  🍽️ meals" if row["meal_included"] else "")
        lines.append(f"• **{row['hostel_name']}** — PKR {int(row['single_room_price']):,}/mo  ⭐{row['overall_rating']}{extras}")
    return {"type":"text","message":"\n".join(lines)}


def respond_location_info(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    max_dist    = entities.get("max_distance_km", 2.0)
    area        = entities.get("area")
    gender      = entities.get("gender")

    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) == 0:
            return {"type":"text","message":f"❓ Couldn't find '{hostel_name}'."}
        row      = row.iloc[0]
        walk_min = round(row["distance_from_fast_km"] / 5.0 * 60)
        return {"type":"text","message":(
            f"**{row['hostel_name']}** location:\n\n"
            f"📍 Area: **{row['area']}**, Islamabad\n"
            f"📏 Distance from FAST H-11: **{row['distance_from_fast_km']} km**\n"
            f"🚌 Transport nearby: **{'Yes ✅' if row['transport_nearby'] else 'No ❌'}**\n"
            f"🚶 Walking time: ~**{walk_min} min**\n"
            f"🗺️ GPS: `{row['latitude']:.4f}, {row['longitude']:.4f}`"
        )}

    df = hostels_df.copy()
    if gender: df = df[df["hostel_type"] == ("Girls" if gender=="Female" else "Boys")]
    if area:   df = df[df["area"].str.lower().str.contains(area.lower(), na=False)]
    df = df[df["distance_from_fast_km"] <= max_dist].sort_values("distance_from_fast_km")
    if len(df) == 0:
        return {"type":"text","message":f"😔 No hostels within {max_dist}km. Try a larger distance."}
    lines = [f"**{len(df)} hostels within {max_dist}km** of FAST H-11:\n"]
    for _, row in df.head(8).iterrows():
        lines.append(f"• **{row['hostel_name']}** — {row['distance_from_fast_km']}km  ({row['area']})  ⭐{row['overall_rating']}")
    return {"type":"text","message":"\n".join(lines)}


def respond_booking_process(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    msg = """**How to Book a Hostel via StayBuddy:**

1️⃣ **Find your hostel** — use Find My Hostel or ask me
2️⃣ **Note the warden contact** — shown in Full Details
3️⃣ **Call or WhatsApp** the warden to confirm availability
4️⃣ **Visit the hostel** before paying any advance
5️⃣ **Documents needed:** CNIC copy, student ID, parent CNIC, 2 passport photos
6️⃣ **Typical advance:** 1–2 months rent as security deposit

📌 Ask for a receipt for every payment.
📌 Confirm cancellation policy before signing anything."""
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) > 0:
            msg += f"\n\n📞 **{row.iloc[0]['hostel_name']} Warden:** `{row.iloc[0]['warden_contact_phone']}`"
    return {"type":"text","message": msg}


def respond_complaint(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    msg = """**Complaint Escalation Steps:**

1️⃣ Talk to the **warden directly** first (48hr window)
2️⃣ If unresolved → escalate to **hostel owner/management**
3️⃣ **Document everything** — photos, dates, times
4️⃣ Common issues:
   • WiFi/electricity → request written maintenance ticket
   • Safety concern → note warden response time
   • Food quality → raise in hostel committee
   • Hygiene → escalate with photos

⚠️ For urgent safety issues contact FAST's student affairs office."""
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) > 0:
            msg += (f"\n\n📞 **{row.iloc[0]['hostel_name']} Warden:** `{row.iloc[0]['warden_contact_phone']}`"
                    f"\n⭐ Warden responsiveness: **{row.iloc[0]['warden_responsiveness']}/5**")
    return {"type":"text","message": msg}


def respond_general_info(entities, hostels_df):
    hostel_name = entities.get("hostel_name")
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name]
        if len(row) > 0:
            row    = row.iloc[0]
            curfew = f"{int(row['curfew_hour']):02d}:00" if row["curfew_hour"] > 0 else "No curfew"
            return {"type":"text","message":(
                f"**{row['hostel_name']}** — General Info:\n\n"
                f"🏠 Type: **{row['hostel_type']} hostel**\n"
                f"📍 Area: **{row['area']}**, Islamabad\n"
                f"✅ Verified: **{'Yes' if row['verified'] else 'No'}**\n"
                f"⭐ Overall: **{row['overall_rating']}/5** ({row['total_reviews']} reviews)\n"
                f"🔒 Security: **{row['security_rating']}/5**\n"
                f"🧹 Cleanliness: **{row['cleanliness_rating']}/5**\n"
                f"📚 Study environment: **{row['study_environment_score']}/1.0**\n"
                f"🕐 Curfew: **{curfew}**\n"
                f"👥 Capacity: **{row['capacity']} students**\n"
                f"📞 Warden: `{row['warden_contact_phone']}`"
            )}
    return {"type":"text","message":"""**General Hostel Information:**

📋 **Check before booking:** verified status, curfew time, noise level, warden responsiveness, electricity & meals inclusion.

🏠 **StayBuddy covers:** 38 Girls · 37 Boys hostels in Islamabad.
💰 **Price range:** PKR 8,152 – 39,833/month
📍 **Areas:** G-13, H-11, F-10, E-11, I-8, I-10, Bahria Town, DHA, Gulberg and more.

💡 Ask about a specific hostel for full details."""}


def respond_low_confidence(intent, confidence, all_scores):
    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:2]
    opts = " or ".join(f"_{s[0].replace('_',' ')}_ ({s[1]:.0%})" for s in sorted_scores)
    return {
        "type":"clarification",
        "message":(
            f"🤔 I'm not quite sure ({confidence:.0%} confidence). Are you asking about {opts}?\n\n"
            f"Try:\n"
            f'• _"Show me girls hostels under 15k near FAST"_\n'
            f'• _"How many girls hostels are there?"_\n'
            f'• _"Which of these have AC?"_ (after a search)\n'
            f'• _"How do I book a hostel?"_'
        ),
    }


# ══════════════════════════════════════════════════════════════════
# MAIN CHAT FUNCTION
# ══════════════════════════════════════════════════════════════════

def chat(
    user_text: str,
    context: ConversationContext,
    tokenizer, model, le,
    nlp,
    hostels_df: pd.DataFrame,
    rec_fn=None,
) -> dict:
    context.turn_count += 1

    # 1. Extract entities
    raw_entities = extract_entities(user_text, nlp, hostels_df)
    context.update(raw_entities)
    entities = context.resolve(raw_entities)

    # 2. Rule-based pre-classification (beats DistilBERT for these cases)
    if is_followup(user_text) and context.last_hostels:
        response = respond_followup(user_text, context, hostels_df, entities)
        intent, confidence, all_scores = "followup", 1.0, {"followup": 1.0}

    elif is_stats_query(user_text):
        response = respond_stats(user_text, entities, hostels_df)
        intent, confidence, all_scores = "stats_query", 1.0, {"stats_query": 1.0}

    else:
        # 3. DistilBERT classification
        intent, confidence, all_scores = classify_intent(user_text, tokenizer, model, le)
        context.last_intent = intent

        if confidence < CONFIDENCE_THRESHOLD:
            response = respond_low_confidence(intent, confidence, all_scores)
        elif intent == "hostel_search":
            response = respond_hostel_search(entities, hostels_df, rec_fn)
        elif intent == "amenity_inquiry":
            response = respond_amenity_inquiry(entities, hostels_df)
        elif intent == "pricing_info":
            response = respond_pricing_info(entities, hostels_df)
        elif intent == "location_info":
            response = respond_location_info(entities, hostels_df)
        elif intent == "booking_process":
            response = respond_booking_process(entities, hostels_df)
        elif intent == "complaint":
            response = respond_complaint(entities, hostels_df)
        elif intent == "general_info":
            response = respond_general_info(entities, hostels_df)
        else:
            response = respond_low_confidence(intent, confidence, all_scores)

    # 4. Store results for follow-up
    if response.get("hostels"):
        context.set_last_hostels(response["hostels"])
    if response.get("names"):
        context.last_hostel_names = response["names"]

    # 5. Attach metadata
    response["intent"]     = intent
    response["confidence"] = confidence
    response["all_scores"] = all_scores
    response["entities"]   = raw_entities
    response["emoji"]      = INTENT_EMOJI.get(intent, "💬")

    return response


# ══════════════════════════════════════════════════════════════════
# EXPORT UTILITY
# ══════════════════════════════════════════════════════════════════

def export_chat_text(history: list, context: ConversationContext) -> str:
    lines = [
        "=" * 60,
        "  STAYBUDDY CHAT EXPORT",
        f"  Date: {context.started_at.strftime('%Y-%m-%d %H:%M')}",
        f"  Turns: {context.turn_count}",
        f"  Preferences remembered: {context.summary()}",
        "=" * 60, ""
    ]
    for msg in history:
        role = "YOU" if msg["role"] == "user" else "STAYBUDDY"
        ts   = msg.get("timestamp", "")
        lines.append(f"[{ts}] {role}:")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)
