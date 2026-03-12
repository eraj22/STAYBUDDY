"""
╔══════════════════════════════════════════════════════════════════════╗
║  StayBuddy — Intelligent Chatbot                                     ║
║  Author : Samiya Saleem (22I-1065)                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  3-LAYER PIPELINE                                                    ║
║                                                                      ║
║  LAYER 1 — UNDERSTANDING                                             ║
║    • DistilBERT (fine-tuned, 7 intents, 84.75% acc)                  ║
║    • spaCy + regex  →  entity extraction                             ║
║    • Rule detectors →  follow-ups & counting queries                 ║
║                                                                      ║
║  LAYER 2 — REASONING                                                 ║
║    • ConversationContext  →  full multi-turn memory                  ║
║    • Confidence threshold →  asks clarification if unsure            ║
║    • Intent router        →  hostel_search fires hybrid engine       ║
║                                                                      ║
║  LAYER 3 — RESPONSE GENERATION                                       ║
║    • Structured handlers  →  live CSV / hybrid engine data           ║
║    • Ollama (llama3)      →  natural language wrapping               ║
║    • Template fallback    →  if Ollama not running                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import re
import os
import joblib
import requests
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

try:
    import spacy
    _SPACY_OK = True
except ImportError:
    _SPACY_OK = False

# ── Path resolution (handles OneDrive spaces in path) ─────────────────
BASE_DIR    = Path(__file__).parent.resolve()
MODEL_DIR   = BASE_DIR / "intent_model"
LABEL_PATH  = BASE_DIR / "label_encoder.pkl"

_CLEAN = Path("C:/staybuddy_models")
if " " in str(BASE_DIR) and _CLEAN.exists():
    MODEL_DIR  = _CLEAN / "intent_model"
    LABEL_PATH = _CLEAN / "label_encoder.pkl"

# ── Tuning knobs ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.55   # below this → ask clarification
MAX_RESULTS          = 5
OLLAMA_URL           = "http://localhost:11434/api/generate"
OLLAMA_MODEL         = "llama3"   # change to "llama3.2" or "mistral" if needed
OLLAMA_TIMEOUT       = 12         # seconds

# ══════════════════════════════════════════════════════════════════════
# LOOKUP TABLES
# ══════════════════════════════════════════════════════════════════════

AMENITY_MAP = {
    # keyword in user text  →  CSV column name
    "wifi":              "has_wifi",
    "internet":          "has_wifi",
    "gym":               "has_gym",
    "fitness":           "has_gym",
    "study room":        "has_study_room",
    "study area":        "has_study_room",
    "library":           "has_library",
    "cafeteria":         "has_cafeteria",
    "canteen":           "has_cafeteria",
    "laundry":           "has_laundry",
    "washing":           "has_laundry",
    "ac":                "has_ac",
    "air conditioning":  "has_ac",
    "air conditioner":   "has_ac",
    "hot water":         "has_hot_water",
    "geyser":            "has_hot_water",
    "generator":         "has_generator",
    "backup":            "has_generator",
    "parking":           "has_parking",
    "prayer room":       "has_prayer_room",
    "mosque":            "has_prayer_room",
    "cctv":              "has_cctv",
    "security guard":    "has_security_guard",
    "guard":             "has_security_guard",
    "common room":       "has_common_room",
    "transport":         "transport_nearby",
}

AMENITY_LABELS = {
    "has_wifi":           "WiFi",
    "has_gym":            "Gym",
    "has_study_room":     "Study Room",
    "has_library":        "Library",
    "has_cafeteria":      "Cafeteria",
    "has_laundry":        "Laundry",
    "has_ac":             "AC",
    "has_hot_water":      "Hot Water",
    "has_generator":      "Generator",
    "has_parking":        "Parking",
    "has_prayer_room":    "Prayer Room",
    "has_cctv":           "CCTV",
    "has_security_guard": "Security Guard",
    "has_common_room":    "Common Room",
    "transport_nearby":   "Transport",
}

INTENT_EMOJI = {
    "hostel_search":   "🔍",
    "amenity_inquiry": "🏷️",
    "pricing_info":    "💰",
    "booking_process": "📋",
    "location_info":   "📍",
    "complaint":       "⚠️",
    "general_info":    "ℹ️",
    "followup":        "↩️",
    "stats_query":     "📊",
}

URDU_BUDGET = {
    "5k":  5000,  "8k":  8000,  "10k": 10000, "12k": 12000,
    "15k": 15000, "20k": 20000, "25k": 25000, "30k": 30000,
    "paanch hazar":    5000,  "aath hazar":      8000,
    "das hazar":       10000, "barah hazar":     12000,
    "pandarah hazar":  15000, "bees hazar":      20000,
}

# multi-word amenities must be checked before single-word ones
MULTI_WORD_AMENITIES = [
    "study room", "study area", "hot water", "air conditioning",
    "air conditioner", "prayer room", "common room", "security guard",
]

AREA_MAP = {
    "g-13": "G-13", "g13": "G-13", "g-14": "G-14", "g14": "G-14",
    "h-11": "H-11", "h11": "H-11", "f-10": "F-10", "f10": "F-10",
    "e-11": "E-11", "e11": "E-11", "i-8":  "I-8",  "i8":  "I-8",
    "i-10": "I-10", "i10": "I-10", "bahria": "Bahria Town",
    "dha":  "DHA Phase 2", "gulberg": "Gulberg", "pwd": "PWD",
}

FOLLOWUP_PATTERNS = [
    r'\b(which|what).*(of these|of them|from these)',
    r'\b(how many|kitne|kitni).*(of these|of them|from these)',
    r'\b(show me|give me).*(cheapest|closest|nearest|best|highest|lowest).*(one|from these)',
    r'\b(cheapest|closest|nearest|best rated|highest rated|safest)\s*(one|hostel)?\s*$',
    r'\b(do any|does any|do they|any of these)\b',
    r'\b(filter|narrow down|from those|of those)\b',
    r'\bwhich one\b',
    # Budget follow-ups on last results
    r'\b(within|under|below|less than).{0,15}(budget|rs|pkr|rupees|\d{4,6})',
    r'\b(are within|within my).{0,20}budget',
    r'\bwhich of these.{0,30}(budget|afford|price|cost)',
]

STATS_PATTERNS = [
    r'\bhow many\b', r'\bkitne\b', r'\bkitni\b',
    r'\btotal (hostels|count|number)\b', r'\bcount\b',
]


# ══════════════════════════════════════════════════════════════════════
# LAYER 1A — DISTILBERT INTENT CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

def load_intent_model():
    """Load fine-tuned DistilBERT + label encoder. Called once via @st.cache_resource."""
    tokenizer = DistilBertTokenizer.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model = DistilBertForSequenceClassification.from_pretrained(
        str(MODEL_DIR), local_files_only=True
    )
    model.eval()
    le = joblib.load(str(LABEL_PATH))
    return tokenizer, model, le


def load_spacy():
    if not _SPACY_OK:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


def _classify(text: str, tokenizer, model, le):
    """
    Run DistilBERT forward pass.
    Returns (intent_str, confidence_float, {intent: prob} dict)
    """
    enc = tokenizer(
        text, max_length=64, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"]
        ).logits
    probs      = torch.softmax(logits, dim=1).squeeze().numpy()
    idx        = int(np.argmax(probs))
    intent     = le.classes_[idx]
    confidence = float(probs[idx])
    all_scores = {le.classes_[i]: float(probs[i]) for i in range(len(le.classes_))}
    return intent, confidence, all_scores


# ══════════════════════════════════════════════════════════════════════
# LAYER 1B — ENTITY EXTRACTION
# ══════════════════════════════════════════════════════════════════════

def extract_entities(text: str, nlp, hostels_df) -> dict:
    """
    Extract structured slots from free text.
    Strategies (in order): spaCy NER → regex → keyword → CSV name match
    """
    ents = {}
    t    = text.lower()

    # 1. spaCy NER for MONEY & location
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC", "FAC") and "location_ref" not in ents:
                ents["location_ref"] = ent.text
            if ent.label_ == "MONEY":
                digits = re.sub(r"[^\d]", "", ent.text)
                if digits and 1000 < int(digits) < 100000:
                    ents["budget"] = int(digits)

    # 2. Budget from regex (PKR / rs / numbers)
    if "budget" not in ents:
        m = re.search(
            r'(?:under|below|less than|upto|up to|within|max|budget.{0,8})?\s*'
            r'(?:rs\.?|pkr|rupees?)?\s*(\d[\d,]+)\s*(?:rs\.?|pkr|rupees?)?',
            t
        )
        if m:
            val = int(m.group(1).replace(",", ""))
            if 1000 < val < 100000:
                ents["budget"] = val
        # Urdu / short-hand numbers
        for word, val in URDU_BUDGET.items():
            if word in t:
                ents["budget"] = val
                break

    # 3. Gender
    if re.search(r'\b(girls?|female|women|ladies|larkiyon|larki)\b', t):
        ents["gender"] = "Female"
    elif re.search(r'\b(boys?|male|men|gents|larkon|larka)\b', t):
        ents["gender"] = "Male"

    # 4. Room type
    rm = re.search(r'\b(single|double|dorm|dormitory|shared)\b', t)
    if rm:
        v = rm.group(1)
        ents["room_type"] = "Dormitory" if v in ("dorm", "dormitory", "shared") else v.capitalize()

    # 5. Distance
    dm = re.search(r'within\s*(\d+\.?\d*)\s*km', t)
    if dm:
        ents["max_distance_km"] = float(dm.group(1))

    # 6. Amenities (multi-word first to avoid partial matches)
    found = []
    for kw in MULTI_WORD_AMENITIES:
        if kw in t and AMENITY_MAP.get(kw) not in found:
            found.append(AMENITY_MAP[kw])
    for kw, col in AMENITY_MAP.items():
        if kw in MULTI_WORD_AMENITIES:
            continue
        if re.search(r'\b' + re.escape(kw) + r'\b', t) and col not in found:
            found.append(col)
    if found:
        ents["amenities"] = found

    # 7. Area / sector
    for key, label in AREA_MAP.items():
        if re.search(r'\b' + re.escape(key) + r'\b', t):
            ents["area"] = label
            break

    # 8. Hostel name from CSV
    if hostels_df is not None:
        for name in hostels_df["hostel_name"].tolist():
            if name.lower() in t:
                ents["hostel_name"] = name
                break

    return ents


# ══════════════════════════════════════════════════════════════════════
# LAYER 1C — RULE-BASED PRE-CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════

RESULT_COMPLAINT_PATTERNS = [
    r'\bi said\b',
    r'\bwhy are you showing\b',
    r'\bthis is (wrong|incorrect|over|too expensive)\b',
    r"\bthat('s| is) (over|above|more than) my budget\b",
    r'\bi (asked|said|mentioned|told you).{0,30}budget\b',
    r'\bignoring my budget\b',
    r'\bnot within my budget\b',
    r'\bover budget\b',
]


def _is_followup(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in FOLLOWUP_PATTERNS)


def _is_stats(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in STATS_PATTERNS)


def _is_result_complaint(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in RESULT_COMPLAINT_PATTERNS)


# ══════════════════════════════════════════════════════════════════════
# LAYER 2 — CONVERSATION CONTEXT  (persisted in st.session_state)
# ══════════════════════════════════════════════════════════════════════

class ConversationContext:
    """
    Remembers every preference across the entire conversation.
    Used to fill in missing slots when user asks follow-up questions
    without repeating themselves.
    """
    def __init__(self):
        self.gender          = None
        self.budget          = None
        self.room_type       = None
        self.max_distance_km = None
        self.amenities       = []
        self.area            = None
        self.last_intent     = None
        self.last_hostels    = []   # list of card dicts from last search result
        self.turn_count      = 0
        self.history         = []   # [{role, text}] for Ollama multi-turn context

    # ── Slot management ────────────────────────────────────────────
    def update(self, entities: dict):
        """Store any newly mentioned preferences."""
        if entities.get("gender"):          self.gender          = entities["gender"]
        if entities.get("budget"):          self.budget          = entities["budget"]
        if entities.get("room_type"):       self.room_type       = entities["room_type"]
        if entities.get("max_distance_km"): self.max_distance_km = entities["max_distance_km"]
        if entities.get("area"):            self.area            = entities["area"]
        for a in entities.get("amenities", []):
            if a not in self.amenities:
                self.amenities.append(a)

    def resolve(self, entities: dict) -> dict:
        """Merge current-turn entities with remembered context."""
        m = dict(entities)
        if not m.get("gender")          and self.gender:          m["gender"]          = self.gender
        if not m.get("budget")          and self.budget:          m["budget"]          = self.budget
        if not m.get("room_type")       and self.room_type:       m["room_type"]       = self.room_type
        if not m.get("max_distance_km") and self.max_distance_km: m["max_distance_km"] = self.max_distance_km
        if not m.get("amenities")       and self.amenities:       m["amenities"]       = list(self.amenities)
        return m

    def set_last_hostels(self, cards: list):
        self.last_hostels = cards

    # ── Ollama history ─────────────────────────────────────────────
    def add_turn(self, role: str, text: str):
        self.history.append({"role": role, "text": text})
        if len(self.history) > 20:
            self.history = self.history[-20:]

    # ── Summary for UI banner ─────────────────────────────────────
    def summary(self) -> str:
        parts = []
        if self.gender:          parts.append(f"Gender: {self.gender}")
        if self.budget:          parts.append(f"Budget: PKR {self.budget:,}")
        if self.room_type:       parts.append(f"Room: {self.room_type}")
        if self.max_distance_km: parts.append(f"Max dist: {self.max_distance_km}km")
        if self.amenities:
            parts.append("Amenities: " + ", ".join(
                AMENITY_LABELS.get(a, a) for a in self.amenities
            ))
        if self.last_hostels:    parts.append(f"Last shown: {len(self.last_hostels)} hostels")
        return " · ".join(parts) if parts else "No preferences stored yet"

    def clear(self):
        self.__init__()


# ══════════════════════════════════════════════════════════════════════
# LAYER 3A — OLLAMA  (natural language generation)
# ══════════════════════════════════════════════════════════════════════

def _ollama_alive() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _ollama_call(prompt: str, system: str) -> str:
    """
    Call local Ollama instance.  Returns '' on any failure so callers
    can fall back to template responses gracefully.
    """
    try:
        payload = {
            "model":  OLLAMA_MODEL,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.35, "top_p": 0.9, "num_predict": 280},
        }
        r = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return ""


def _system_prompt(context: ConversationContext, hostels_df) -> str:
    """Build the Ollama system prompt with live dataset facts + user profile."""
    g = len(hostels_df[hostels_df["hostel_type"] == "Girls"]) if hostels_df is not None else 38
    b = len(hostels_df[hostels_df["hostel_type"] == "Boys"])  if hostels_df is not None else 37
    pmin = int(hostels_df["single_room_price"].min()) if hostels_df is not None else 8152
    pmax = int(hostels_df["single_room_price"].max()) if hostels_df is not None else 39833

    last = ""
    if context.last_hostels:
        names = [h["name"] for h in context.last_hostels[:5]]
        last  = "\nLast recommended: " + ", ".join(names)

    return f"""You are the StayBuddy assistant — a helpful, friendly chatbot for students at FAST NUCES Islamabad searching for hostels.

LIVE DATASET FACTS:
- {g + b} hostels total: {g} girls, {b} boys
- Price range: PKR {pmin:,} – {pmax:,}/month
- All hostels are near FAST NUCES H-11 campus, Islamabad
- Areas: G-13, H-11, F-10, E-11, I-8, I-10, Bahria Town, DHA, Gulberg{last}

USER PROFILE SO FAR:
{context.summary()}

RULES:
- Be concise and friendly — like a helpful senior student
- Respond in the same language as the user (Urdu or English)
- Never invent hostel names, prices, or features
- If unsure about something specific, say so honestly
- Keep replies under 160 words
- Do NOT repeat the user's question back to them"""


def _history_for_ollama(context: ConversationContext, new_msg: str) -> str:
    """Format last few turns as a readable string for the Ollama prompt."""
    lines = []
    for t in context.history[-6:]:
        role = "Student" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['text']}")
    lines.append(f"Student: {new_msg}")
    return "\n".join(lines)


def _enhance(user_text: str, structured: dict, context: ConversationContext,
             hostels_df, intent: str, entities: dict) -> str:
    """
    Use Ollama to wrap structured data in natural language.
    For hostel_results: write the intro sentence only (cards stay separate).
    For text responses: rewrite in conversational tone, keeping all facts.
    Falls back to the structured message if Ollama returns nothing.
    """
    if structured.get("type") == "hostel_results":
        cards = structured.get("hostels", [])
        if not cards:
            return structured["message"]
        hostel_lines = "\n".join(
            f"- {h['name']}: PKR {h['price']:,}/mo, "
            f"{h['distance']}km from FAST, rated {h['rating']}/5, "
            f"amenities: {', '.join(h['amenities'][:4])}"
            for h in cards
        )
        filters = f"Gender: {entities.get('gender', context.gender or 'any')}"
        if entities.get("budget"):
            filters += f", Budget ≤ PKR {entities['budget']:,}"
        prompt = (
            f"Conversation:\n{_history_for_ollama(context, user_text)}\n\n"
            f"Filters: {filters}\n"
            f"Results:\n{hostel_lines}\n\n"
            f"Write ONE short, friendly opening line (max 20 words) introducing "
            f"these results. Do not list the hostels."
        )
    else:
        # Only enhance for intents where tone matters
        if intent not in ("booking_process", "complaint", "general_info",
                          "location_info", "pricing_info", "amenity_inquiry",
                          "stats_query"):
            return structured["message"]
        prompt = (
            f"Conversation:\n{_history_for_ollama(context, user_text)}\n\n"
            f"Factual answer:\n{structured['message']}\n\n"
            f"Rewrite as a natural, conversational reply. "
            f"Keep every fact and number exactly as given. "
            f"Do NOT add new information. Under 160 words."
        )

    reply = _ollama_call(prompt, _system_prompt(context, hostels_df))
    return reply if reply else structured["message"]


# ══════════════════════════════════════════════════════════════════════
# LAYER 3B — STRUCTURED DATA HANDLERS
# ══════════════════════════════════════════════════════════════════════

def _make_card(row) -> dict:
    return {
        "name":      row["hostel_name"],
        "type":      row["hostel_type"],
        "area":      row["area"],
        "price":     int(row["single_room_price"]),
        "rating":    row["overall_rating"],
        "distance":  row["distance_from_fast_km"],
        "security":  row["security_rating"],
        "elec_inc":  bool(row.get("electricity_included", 0)),
        "meal_inc":  bool(row.get("meal_included", 0)),
        "transport": bool(row.get("transport_nearby", 0)),
        "curfew":    row.get("curfew_hour", 0),
        "amenities": [
            AMENITY_LABELS[c] for c in AMENITY_LABELS
            if row.get(c, 0) == 1
        ][:6],
    }


# ── Hostel search ──────────────────────────────────────────────────────
def _handle_search(entities: dict, hostels_df, rec_fn) -> dict:
    gender    = entities.get("gender", "Male")
    budget    = entities.get("budget")
    max_dist  = entities.get("max_distance_km", 5.0)
    amenities = entities.get("amenities", [])
    room_type = entities.get("room_type", "Single")
    area      = entities.get("area")
    htype     = "Girls" if gender == "Female" else "Boys"

    # Try hybrid engine
    cards       = []
    used_engine = False
    if rec_fn is not None:
        try:
            must_have = [AMENITY_LABELS.get(a, a) for a in amenities]
            recs      = rec_fn(
                gender=gender, department="Computer Science",
                budget_max=budget or 25000, max_dist=max_dist,
                study_pref=0.6, food_pref="Both", room_type=room_type,
                price_sens=0.6, comfort_pref=0.5, noise_tol=0.3,
                curfew_flex=0.5, needs_transport=(max_dist > 3.0),
                must_have=must_have, top_k=MAX_RESULTS,
            )
            if recs is not None and len(recs) > 0:
                # Hard enforce budget AFTER hybrid engine returns results
                if budget:
                    recs = recs[recs["single_room_price"] <= budget]
                used_engine = True
                for _, row in recs.iterrows():
                    c = _make_card(row)
                    if "hybrid_score" in row:
                        c["score"] = round(float(row["hybrid_score"]) * 100, 1)
                    cards.append(c)
        except Exception:
            pass

    # Fallback: direct CSV filter
    if not cards:
        df = hostels_df[hostels_df["hostel_type"] == htype].copy()
        if budget:   df = df[df["single_room_price"] <= budget]
        if max_dist: df = df[df["distance_from_fast_km"] <= max_dist]
        for a in amenities:
            if a in df.columns:
                df = df[df[a] == 1]
        if area:
            df = df[df["area"].str.lower().str.contains(area.lower(), na=False)]
        df["_s"] = (
            df["overall_rating"] / 5.0
            + (1 - df["distance_from_fast_km"] / 6.14) * 0.2
        )
        if budget:
            df["_s"] += (1 - df["single_room_price"] / budget).clip(0, 1) * 0.3
        df = df.sort_values("_s", ascending=False).head(MAX_RESULTS)
        for _, row in df.iterrows():
            cards.append(_make_card(row))

    if not cards:
        return {
            "type": "text",
            "message": f"😔 No {htype.lower()} hostels matched your criteria. Try relaxing budget or distance.",
        }

    # Build summary message
    filter_parts = []
    if budget:                            filter_parts.append(f"≤ PKR {budget:,}")
    if max_dist and max_dist != 5.0:      filter_parts.append(f"≤ {max_dist}km")
    if amenities:
        filter_parts.append(", ".join(AMENITY_LABELS.get(a, a) for a in amenities))
    msg = f"Found **{len(cards)}** {htype} hostels"
    if filter_parts:
        msg += " · " + " · ".join(filter_parts)

    return {
        "type":        "hostel_results",
        "message":     msg,
        "hostels":     cards,
        "used_engine": used_engine,
        "note": (
            "🧠 _Powered by Eraj's hybrid recommendation engine_"
            if used_engine else "📊 _Filtered from live CSV data_"
        ),
    }


# ── Follow-up: filter / sort the last shown results ────────────────────
def _handle_followup(text: str, context: ConversationContext,
                     hostels_df, entities: dict):
    t = text.lower()
    n = len(context.last_hostels)
    if n == 0:
        return None

    names = [h["name"] for h in context.last_hostels]
    df    = hostels_df[hostels_df["hostel_name"].isin(names)].copy()

    def _cards(sub):
        out = []
        for _, row in sub.iterrows():
            out.append(_make_card(row))
        return out

    # Filter by amenity
    amenities = entities.get("amenities", [])
    if amenities:
        filtered = df.copy()
        for a in amenities:
            if a in filtered.columns:
                filtered = filtered[filtered[a] == 1]
        lbl   = " + ".join(AMENITY_LABELS.get(a, a) for a in amenities)
        count = len(filtered)
        msg   = (
            f"**{count} of the {n} hostels** have {lbl}:"
            if count > 0 else f"None of the {n} hostels have {lbl}."
        )
        cards = _cards(filtered) if count > 0 else []
        context.set_last_hostels(cards)
        return _result("followup", 1.0, entities, msg, cards)

    # Filter by budget
    budget = entities.get("budget") or context.budget
    if budget and re.search(r'budget|afford|under|below|cheap|sasta', t):
        filtered = df[df["single_room_price"] <= budget].sort_values("overall_rating", ascending=False)
        msg = (
            f"**{len(filtered)} of the {n} hostels** are within PKR {budget:,}/mo:"
            if len(filtered) > 0 else f"None within PKR {budget:,}/mo."
        )
        cards = _cards(filtered)
        context.set_last_hostels(cards)
        return _result("followup", 1.0, entities, msg, cards)

    # Superlatives
    if re.search(r'\bcheapest|sasta\b', t):
        row = df.sort_values("single_room_price").iloc[0]
    elif re.search(r'\bclosest|nearest|paas\b', t):
        row = df.sort_values("distance_from_fast_km").iloc[0]
    elif re.search(r'\bbest|highest.rated|top\b', t):
        row = df.sort_values("overall_rating", ascending=False).iloc[0]
    elif re.search(r'\bsafest|most.secure\b', t):
        row = df.sort_values("security_rating", ascending=False).iloc[0]
    elif re.search(r'\bmost.amenities\b', t):
        amenity_cols = [c for c in AMENITY_LABELS if c in df.columns]
        df["_am"] = df[amenity_cols].sum(axis=1)
        row = df.sort_values("_am", ascending=False).iloc[0]
    else:
        return None

    card = _make_card(row)
    context.set_last_hostels([card])
    return _result("followup", 1.0, entities,
                   "From your previous results, here is the best match:", [card])


def _result(intent, conf, entities, msg, hostels=None):
    """Helper: build a standard hostel_results response dict."""
    return {
        "type":        "hostel_results" if hostels is not None else "text",
        "message":     msg,
        "hostels":     hostels or [],
        "used_engine": False,
        "intent":      intent,
        "confidence":  conf,
        "entities":    entities,
        "emoji":       INTENT_EMOJI.get(intent, "💬"),
        "note":        "📊 _Filtered from live CSV data_",
    }


# ── Stats / counting ───────────────────────────────────────────────────
def _handle_stats(text: str, entities: dict, hostels_df) -> dict:
    gender    = entities.get("gender")
    area      = entities.get("area")
    amenities = entities.get("amenities", [])

    df = hostels_df.copy()
    if gender:
        htype = "Girls" if gender == "Female" else "Boys"
        df    = df[df["hostel_type"] == htype]
    for a in amenities:
        if a in df.columns:
            df = df[df[a] == 1]
    if area:
        df = df[df["area"].str.lower().str.contains(area.lower(), na=False)]

    count      = len(df)
    type_label = ("girls" if gender == "Female" else "boys") if gender else "total"

    if amenities:
        lbl = " + ".join(AMENITY_LABELS.get(a, a) for a in amenities)
        msg = f"**{count} {type_label} hostels** have {lbl}."
        if 0 < count <= 8:
            msg += "\n\n" + "\n".join(f"• {r}" for r in df["hostel_name"].tolist())
    elif area:
        msg = f"**{count} {type_label} hostels** are in {area}."
    elif gender:
        msg = f"There are **{count} {type_label} hostels** in StayBuddy."
    else:
        g = len(hostels_df[hostels_df["hostel_type"] == "Girls"])
        b = len(hostels_df[hostels_df["hostel_type"] == "Boys"])
        msg = (
            f"StayBuddy has **{g + b} hostels** in total — "
            f"**{g} girls** and **{b} boys** hostels across Islamabad."
        )

    return {"type": "text", "message": msg,
            "intent": "stats_query", "confidence": 1.0,
            "entities": entities, "emoji": "📊"}


# ── Amenity inquiry ────────────────────────────────────────────────────
def _handle_amenity(entities: dict, hostels_df) -> dict:
    amenities   = entities.get("amenities", [])
    hostel_name = entities.get("hostel_name")
    gender      = entities.get("gender")

    # Specific hostel query
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name.lower()]
        if len(row) == 0:
            return {"type": "text", "message": f"❓ Couldn't find '{hostel_name}'."}
        row = row.iloc[0]
        if amenities:
            lines = [f"**{row['hostel_name']}** — Amenity check:\n"]
            for a in amenities:
                lbl  = AMENITY_LABELS.get(a, a)
                have = bool(row.get(a, 0))
                lines.append(f"{'✅' if have else '❌'} {lbl}: {'Yes' if have else 'No'}")
            return {"type": "text", "message": "\n".join(lines)}
        have = [AMENITY_LABELS[c] for c in AMENITY_LABELS if row.get(c, 0) == 1]
        return {"type": "text",
                "message": f"**{row['hostel_name']}** has: {', '.join(have) or 'None listed'}"}

    # Search all hostels
    if amenities:
        df = hostels_df.copy()
        if gender:
            df = df[df["hostel_type"] == ("Girls" if gender == "Female" else "Boys")]
        for a in amenities:
            if a in df.columns:
                df = df[df[a] == 1]
        df   = df.sort_values("overall_rating", ascending=False)
        lbl  = " + ".join(AMENITY_LABELS.get(a, a) for a in amenities)
        if len(df) == 0:
            return {"type": "text", "message": f"😔 No hostels found with {lbl}."}
        lines = [f"**{len(df)} hostels** have {lbl}:\n"]
        for _, r in df.head(6).iterrows():
            lines.append(
                f"• **{r['hostel_name']}** — {r['area']} · "
                f"⭐{r['overall_rating']} · PKR {int(r['single_room_price']):,}/mo"
            )
        if len(df) > 6:
            lines.append(f"_...and {len(df) - 6} more._")
        return {"type": "text", "message": "\n".join(lines)}

    return {"type": "text",
            "message": "Which amenity are you asking about? (WiFi, gym, AC, study room, etc.)"}


# ── Pricing ────────────────────────────────────────────────────────────
def _handle_pricing(entities: dict, hostels_df) -> dict:
    hostel_name = entities.get("hostel_name")
    budget      = entities.get("budget")
    gender      = entities.get("gender")

    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name.lower()]
        if len(row) == 0:
            return {"type": "text", "message": f"❓ Couldn't find '{hostel_name}'."}
        r     = row.iloc[0]
        lines = [f"**{r['hostel_name']}** — Pricing:\n"]
        lines.append(f"• Single room: **PKR {int(r['single_room_price']):,}/mo**")
        if pd.notna(r.get("double_room_price")) and r.get("double_room_price", 0) > 0:
            lines.append(f"• Double room: PKR {int(r['double_room_price']):,}/mo")
        if pd.notna(r.get("dorm_room_price")) and r.get("dorm_room_price", 0) > 0:
            lines.append(f"• Dorm bed:    PKR {int(r['dorm_room_price']):,}/mo")
        elec_extra = 0 if r["electricity_included"] else 2000
        lines.append(
            f"• Electricity: {'✅ Included' if r['electricity_included'] else '❌ Separate (~PKR 2,000)'}"
        )
        lines.append(f"• Meals: {'✅ Included' if r['meal_included'] else '❌ Not included'}")
        lines.append(f"\n💰 **Estimated total: PKR {int(r['single_room_price']) + elec_extra:,}/mo**")
        return {"type": "text", "message": "\n".join(lines)}

    df = hostels_df.copy()
    if gender:
        df = df[df["hostel_type"] == ("Girls" if gender == "Female" else "Boys")]
    if budget:
        df = df[df["single_room_price"] <= budget]
    df = df.sort_values("single_room_price").head(5)
    if len(df) == 0:
        return {"type": "text", "message": f"😔 No hostels found under PKR {budget:,}."}
    lines = [
        f"**Cheapest options** "
        f"(PKR {int(df['single_room_price'].min()):,} – "
        f"{int(df['single_room_price'].max()):,}/mo):\n"
    ]
    for _, r in df.iterrows():
        extras = "  ".join(filter(None, [
            "⚡ incl." if r["electricity_included"] else "",
            "🍽️ meals"  if r["meal_included"]        else "",
        ]))
        lines.append(
            f"• **{r['hostel_name']}** — PKR {int(r['single_room_price']):,}/mo  "
            f"⭐{r['overall_rating']}  {extras}"
        )
    return {"type": "text", "message": "\n".join(lines)}


# ── Location ────────────────────────────────────────────────────────────
def _handle_location(entities: dict, hostels_df) -> dict:
    hostel_name = entities.get("hostel_name")
    max_dist    = entities.get("max_distance_km", 2.0)
    area        = entities.get("area")
    gender      = entities.get("gender")

    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name.lower()]
        if len(row) == 0:
            return {"type": "text", "message": f"❓ Couldn't find '{hostel_name}'."}
        r        = row.iloc[0]
        walk_min = round(r["distance_from_fast_km"] / 5.0 * 60)
        msg = (
            f"**{r['hostel_name']}** — Location:\n\n"
            f"📍 Area: **{r['area']}**, Islamabad\n"
            f"📏 Distance from FAST H-11: **{r['distance_from_fast_km']} km**\n"
            f"🚌 Transport nearby: **{'Yes ✅' if r['transport_nearby'] else 'No ❌'}**\n"
            f"🚶 Walking estimate: ~**{walk_min} min**\n"
            f"🗺️ GPS: `{r['latitude']:.4f}, {r['longitude']:.4f}`"
        )
        return {"type": "text", "message": msg}

    df = hostels_df.copy()
    if gender:
        df = df[df["hostel_type"] == ("Girls" if gender == "Female" else "Boys")]
    if area:
        df = df[df["area"].str.lower().str.contains(area.lower(), na=False)]
    df = df[df["distance_from_fast_km"] <= max_dist].sort_values("distance_from_fast_km")
    if len(df) == 0:
        return {"type": "text",
                "message": f"😔 No hostels within {max_dist}km. Try increasing the distance."}
    lines = [f"**{len(df)} hostels within {max_dist}km** of FAST H-11:\n"]
    for _, r in df.head(6).iterrows():
        lines.append(
            f"• **{r['hostel_name']}** — {r['distance_from_fast_km']}km "
            f"· {r['area']} · ⭐{r['overall_rating']}"
        )
    return {"type": "text", "message": "\n".join(lines)}


# ── Booking process ─────────────────────────────────────────────────────
def _handle_booking(entities: dict, hostels_df) -> dict:
    hostel_name = entities.get("hostel_name")
    msg = (
        "**How to Book a Hostel via StayBuddy:**\n\n"
        "1️⃣ **Find your hostel** — use Find My Hostel or ask me\n"
        "2️⃣ **Note warden contact** — shown in Full Details\n"
        "3️⃣ **Call or WhatsApp** the warden to confirm availability\n"
        "4️⃣ **Visit first** before paying any advance\n"
        "5️⃣ **Documents needed:** CNIC, student ID, parent CNIC, 2 passport photos\n"
        "6️⃣ **Security deposit:** typically 1–2 months rent\n\n"
        "📌 Always get a receipt.  📌 Confirm cancellation policy in writing."
    )
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name.lower()]
        if len(row) > 0:
            msg += f"\n\n📞 **{row.iloc[0]['hostel_name']} warden:** `{row.iloc[0]['warden_contact_phone']}`"
    return {"type": "text", "message": msg}


# ── Complaint ───────────────────────────────────────────────────────────
def _handle_complaint(entities: dict, hostels_df) -> dict:
    hostel_name = entities.get("hostel_name")
    msg = (
        "**How to raise a complaint:**\n\n"
        "1️⃣ Talk to the warden directly first\n"
        "2️⃣ If unresolved in 48 hours → escalate to hostel owner\n"
        "3️⃣ Document everything — photos, dates, messages\n\n"
        "**Common issues:**\n"
        "• WiFi / electricity → request written maintenance ticket\n"
        "• Safety concern → note warden response time & escalate\n"
        "• Food quality → raise in hostel committee meeting\n\n"
        "⚠️ _For urgent safety issues, contact FAST student affairs immediately._"
    )
    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name.lower()]
        if len(row) > 0:
            r    = row.iloc[0]
            msg += (
                f"\n\n📞 **{r['hostel_name']} warden:** `{r['warden_contact_phone']}`  "
                f"Responsiveness: ⭐{r['warden_responsiveness']}/5"
            )
    return {"type": "text", "message": msg}


# ── General info ────────────────────────────────────────────────────────
def _handle_general(entities: dict, hostels_df) -> dict:
    hostel_name = entities.get("hostel_name")

    if hostel_name:
        row = hostels_df[hostels_df["hostel_name"].str.lower() == hostel_name.lower()]
        if len(row) > 0:
            r      = row.iloc[0]
            curfew = f"{int(r['curfew_hour']):02d}:00" if r["curfew_hour"] > 0 else "No curfew"
            msg = (
                f"**{r['hostel_name']}** — Full Info:\n\n"
                f"🏠 Type: {r['hostel_type']} hostel\n"
                f"📍 Area: {r['area']}, Islamabad\n"
                f"✅ Verified: {'Yes' if r['verified'] else 'No'}\n"
                f"⭐ Rating: {r['overall_rating']}/5 ({r['total_reviews']} reviews)\n"
                f"🔒 Security: {r['security_rating']}/5\n"
                f"🧹 Cleanliness: {r['cleanliness_rating']}/5\n"
                f"📚 Study environment: {r['study_environment_score']}/1.0\n"
                f"🕐 Curfew: {curfew}\n"
                f"👥 Capacity: {r['capacity']} students\n"
                f"📞 Warden: `{r['warden_contact_phone']}`"
            )
            return {"type": "text", "message": msg}

    g    = len(hostels_df[hostels_df["hostel_type"] == "Girls"])
    b    = len(hostels_df[hostels_df["hostel_type"] == "Boys"])
    pmin = int(hostels_df["single_room_price"].min())
    pmax = int(hostels_df["single_room_price"].max())
    msg  = (
        f"**StayBuddy** covers **{g + b} hostels** near FAST NUCES Islamabad.\n\n"
        f"🏠 {g} Girls  ·  {b} Boys\n"
        f"💰 PKR {pmin:,} – {pmax:,}/month\n"
        f"📍 G-13 · H-11 · F-10 · E-11 · I-8 · Bahria Town · DHA · Gulberg\n\n"
        f"💡 Ask about a specific hostel, or use **Find My Hostel** for personalised recommendations."
    )
    return {"type": "text", "message": msg}


# ══════════════════════════════════════════════════════════════════════
# MAIN CHAT FUNCTION  (called by app.py)
# ══════════════════════════════════════════════════════════════════════

def chat(
    user_text: str,
    context: ConversationContext,
    tokenizer, model, le,
    nlp,
    hostels_df: pd.DataFrame,
    rec_fn=None,
) -> dict:
    """
    Full 3-layer pipeline.

    Returns a response dict with keys:
      type          — "text" | "hostel_results" | "clarification"
      message       — string to display
      hostels       — list of card dicts (if type == hostel_results)
      intent        — classified intent string
      confidence    — float 0-1
      entities      — raw extracted entities dict
      emoji         — intent emoji
      used_engine   — bool (True = hybrid engine was used)
      ollama_used   — bool
      note          — source attribution string
    """
    context.turn_count += 1

    # ── Step 1: Extract entities from this turn ────────────────────
    raw_entities = extract_entities(user_text, nlp, hostels_df)
    context.update(raw_entities)            # store in memory
    entities = context.resolve(raw_entities)  # fill gaps from memory

    # ── Step 2: Rule-based pre-classification (beats DistilBERT) ──

    # Follow-up (refers to last shown hostels)
    if _is_followup(user_text) and context.last_hostels:
        result = _handle_followup(user_text, context, hostels_df, entities)
        if result:
            if _ollama_alive():
                result["message"] = _enhance(
                    user_text, result, context, hostels_df, "followup", entities
                )
                result["ollama_used"] = True
            else:
                result["ollama_used"] = False
            context.add_turn("user", user_text)
            context.add_turn("assistant", result["message"])
            return result

    # Stats / counting
    if _is_stats(user_text):
        result = _handle_stats(user_text, entities, hostels_df)
        if _ollama_alive():
            result["message"] = _enhance(
                user_text, result, context, hostels_df, "stats_query", entities
            )
            result["ollama_used"] = True
        else:
            result["ollama_used"] = False
        context.add_turn("user", user_text)
        context.add_turn("assistant", result["message"])
        return result

    # ── Step 2c: Result complaint — redo search with strict filters ─
    if _is_result_complaint(user_text) and context.last_hostels:
        # User is saying the results were wrong — redo with hard budget
        budget = entities.get("budget") or context.budget
        gender = entities.get("gender") or context.gender or "Male"
        htype  = "Girls" if gender == "Female" else "Boys"
        df     = hostels_df[hostels_df["hostel_type"] == htype].copy()
        if budget:
            df = df[df["single_room_price"] <= budget]
        if context.max_distance_km:
            df = df[df["distance_from_fast_km"] <= context.max_distance_km]
        df = df.sort_values("overall_rating", ascending=False).head(MAX_RESULTS)

        if len(df) == 0:
            apology = (
                f"Sorry about that! Unfortunately there are no {htype.lower()} hostels "
                f"within PKR {budget:,}/mo"
                + (f" and {context.max_distance_km}km" if context.max_distance_km else "")
                + ". The minimum price in our dataset is PKR 8,152/mo. "
                "Try increasing your budget slightly."
            )
            result = {"type": "text", "message": apology,
                      "intent": "hostel_search", "confidence": 1.0,
                      "entities": entities, "emoji": "🔍", "ollama_used": False}
        else:
            cards = [_make_card(row) for _, row in df.iterrows()]
            context.set_last_hostels(cards)
            result = {
                "type": "hostel_results",
                "message": (
                    f"Apologies for the confusion! Here are {htype} hostels "
                    f"strictly within PKR {budget:,}/mo:"
                    if budget else "Here are corrected results:"
                ),
                "hostels": cards, "used_engine": False,
                "intent": "hostel_search", "confidence": 1.0,
                "entities": entities, "emoji": "🔍",
                "note": "📊 _Strictly filtered by your budget_",
                "ollama_used": False,
            }
        context.add_turn("user", user_text)
        context.add_turn("assistant", result["message"])
        return result

    # ── Step 3: DistilBERT intent classification ───────────────────
    intent, confidence, all_scores = _classify(user_text, tokenizer, model, le)
    context.last_intent = intent

    # ── Step 4: Confidence gate ────────────────────────────────────
    if confidence < CONFIDENCE_THRESHOLD:
        top2 = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        opts = " or ".join(
            f"_{s[0].replace('_',' ')}_ ({s[1]:.0%})" for s in top2
        )
        result = {
            "type":       "clarification",
            "message": (
                f"🤔 I'm not sure what you mean ({confidence:.0%} confident).\n\n"
                f"Are you asking about {opts}?\n\n"
                f"Try something like:\n"
                f'• _"Show me girls hostels under 15k with WiFi"_\n'
                f'• _"Does Khadija Residence have a gym?"_\n'
                f'• _"How do I book a hostel?"_'
            ),
            "intent":      intent,
            "confidence":  confidence,
            "all_scores":  all_scores,
            "entities":    raw_entities,
            "emoji":       "🤔",
            "ollama_used": False,
        }
        context.add_turn("user", user_text)
        context.add_turn("assistant", result["message"])
        return result

    # ── Step 5: Route to structured handler ───────────────────────
    if   intent == "hostel_search":   structured = _handle_search(entities, hostels_df, rec_fn)
    elif intent == "amenity_inquiry": structured = _handle_amenity(entities, hostels_df)
    elif intent == "pricing_info":    structured = _handle_pricing(entities, hostels_df)
    elif intent == "location_info":   structured = _handle_location(entities, hostels_df)
    elif intent == "booking_process": structured = _handle_booking(entities, hostels_df)
    elif intent == "complaint":       structured = _handle_complaint(entities, hostels_df)
    elif intent == "general_info":    structured = _handle_general(entities, hostels_df)
    else:
        structured = {
            "type":    "text",
            "message": "I'm not sure how to help with that. Try asking about hostels, pricing, amenities, or booking.",
        }

    # After hostel_search, remember the results for follow-ups
    if structured.get("type") == "hostel_results" and structured.get("hostels"):
        context.set_last_hostels(structured["hostels"])

    # ── Step 6: Ollama — wrap in natural language ──────────────────
    if _ollama_alive():
        structured["message"] = _enhance(
            user_text, structured, context, hostels_df, intent, entities
        )
        structured["ollama_used"] = True
    else:
        structured["ollama_used"] = False

    # ── Step 7: Attach metadata ────────────────────────────────────
    structured.setdefault("intent",     intent)
    structured.setdefault("confidence", confidence)
    structured.setdefault("entities",   raw_entities)
    structured.setdefault("emoji",      INTENT_EMOJI.get(intent, "💬"))
    structured["all_scores"] = all_scores

    # ── Step 8: Store in conversation history ──────────────────────
    context.add_turn("user",      user_text)
    context.add_turn("assistant", structured["message"])

    return structured


# ══════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════

def export_chat_text(chat_history: list, context: ConversationContext) -> str:
    """Export the full conversation as a downloadable text file."""
    lines = [
        "StayBuddy — Chat Export",
        f"Exported : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Profile  : {context.summary()}",
        "=" * 52,
        "",
    ]
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "StayBuddy"
        ts   = msg.get("timestamp", "")
        lines.append(f"[{ts}]  {role}:")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)