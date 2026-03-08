import re
import spacy
import pandas as pd

# ── Load spaCy and hostel data ─────────────────────────
nlp        = spacy.load("en_core_web_sm")
hostels_df = pd.read_csv("hostels.csv")
KNOWN_HOSTELS = hostels_df["hostel_name"].str.lower().tolist()

# ── Same maps as chatbot.py ────────────────────────────
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
    "common room":      "has_common_room"
}

URDU_NUMBERS = {
    "10k": 10000, "12k": 12000, "15k": 15000,
    "20k": 20000, "8k":  8000,  "5k":  5000,
    "das hazar":     10000, "barah hazar":    12000,
    "pandarah hazar": 15000, "paanch hazar":   5000
}

# ── Entity extractor (same logic as chatbot.py) ────────
def extract_entities(text):
    entities   = {}
    text_lower = text.lower()
    doc        = nlp(text)

    # spaCy named entities
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC", "FAC"]:
            entities["location_ref"] = ent.text
        if ent.label_ == "MONEY":
            amount = re.sub(r"[^\d]", "", ent.text)
            if amount and int(amount) > 1000:
                entities["budget"] = int(amount)

    # Budget regex fallback
    if "budget" not in entities:
        match = re.search(
            r'(\d[\d,]*)\s*(rupees?|rs\.?|pkr)?', text_lower)
        if match:
            val = int(match.group(1).replace(",", ""))
            if val > 1000:
                entities["budget"] = val
        for word, value in URDU_NUMBERS.items():
            if word in text_lower:
                entities["budget"] = value
                break

    # Room type
    room = re.search(
        r'\b(single|double|dorm|shared)\b', text_lower)
    if room:
        entities["room_type"] = room.group(1)

    # Distance
    dist = re.search(
        r'within\s*(\d+\.?\d*)\s*km', text_lower)
    if dist:
        entities["max_distance_km"] = float(dist.group(1))

    # Amenities
    found = []
    multi = ["study room", "study area", "hot water",
             "air conditioning", "prayer room",
             "common room", "security guard"]
    for kw in multi:
        if kw in text_lower:
            found.append(AMENITY_MAP[kw])
    for kw, col in AMENITY_MAP.items():
        if kw not in multi:
            if re.search(r'\b' + re.escape(kw) + r'\b',
                         text_lower):
                if col not in found:
                    found.append(col)
    if found:
        entities["amenities"] = list(set(found))

    # Hostel name
    for name in KNOWN_HOSTELS:
        if name in text_lower:
            entities["hostel_name"] = name
            break

    return entities

# ─────────────────────────────────────────────────────
# TEST CASES — each has input text and expected entities
# ─────────────────────────────────────────────────────
TEST_CASES = [

    # ── BUDGET EXTRACTION ──────────────────────────────
    {
        "category": "budget",
        "text":     "Show me hostels under 15000 rupees",
        "expected": {"budget": 15000}
    },
    {
        "category": "budget",
        "text":     "I need a hostel under 12000",
        "expected": {"budget": 12000}
    },
    {
        "category": "budget",
        "text":     "koi hostel hai 10k mein",
        "expected": {"budget": 10000}
    },
    {
        "category": "budget",
        "text":     "budget is Rs. 8,000 per month",
        "expected": {"budget": 8000}
    },
    {
        "category": "budget",
        "text":     "affordable hostel under 20k",
        "expected": {"budget": 20000}
    },
    {
        "category": "budget",
        "text":     "das hazar mein hostel chahiye",
        "expected": {"budget": 10000}
    },

    # ── AMENITY EXTRACTION ─────────────────────────────
    {
        "category": "amenity",
        "text":     "Does this hostel have WiFi",
        "expected": {"amenities": ["has_wifi"]}
    },
    {
        "category": "amenity",
        "text":     "I need a hostel with gym and study room",
        "expected": {"amenities": ["has_gym", "has_study_room"]}
    },
    {
        "category": "amenity",
        "text":     "Is there a prayer room available",
        "expected": {"amenities": ["has_prayer_room"]}
    },
    {
        "category": "amenity",
        "text":     "hostel with generator and hot water",
        "expected": {"amenities": ["has_generator", "has_hot_water"]}
    },
    {
        "category": "amenity",
        "text":     "is laundry service available",
        "expected": {"amenities": ["has_laundry"]}
    },
    {
        "category": "amenity",
        "text":     "I want AC and parking facility",
        "expected": {"amenities": ["has_ac", "has_parking"]}
    },

    # ── HOSTEL NAME EXTRACTION ─────────────────────────
    {
        "category": "hostel_name",
        "text":     f"How much is a room at {hostels_df['hostel_name'].iloc[0]}",
        "expected": {"hostel_name":
                     hostels_df["hostel_name"].iloc[0].lower()}
    },
    {
        "category": "hostel_name",
        "text":     f"Does {hostels_df['hostel_name'].iloc[1]} have a gym",
        "expected": {"hostel_name":
                     hostels_df["hostel_name"].iloc[1].lower()}
    },
    {
        "category": "hostel_name",
        "text":     f"How far is {hostels_df['hostel_name'].iloc[2]} from FAST",
        "expected": {"hostel_name":
                     hostels_df["hostel_name"].iloc[2].lower()}
    },
    {
        "category": "hostel_name",
        "text":     f"What are the prices at "
                    f"{hostels_df['hostel_name'].iloc[3]}",
        "expected": {"hostel_name":
                     hostels_df["hostel_name"].iloc[3].lower()}
    },
    {
        "category": "hostel_name",
        "text":     f"Is {hostels_df['hostel_name'].iloc[4]} "
                    f"near campus",
        "expected": {"hostel_name":
                     hostels_df["hostel_name"].iloc[4].lower()}
    },

    # ── ROOM TYPE EXTRACTION ───────────────────────────
    {
        "category": "room_type",
        "text":     "I want a single room hostel",
        "expected": {"room_type": "single"}
    },
    {
        "category": "room_type",
        "text":     "looking for a double room",
        "expected": {"room_type": "double"}
    },
    {
        "category": "room_type",
        "text":     "how much is a dorm bed",
        "expected": {"room_type": "dorm"}
    },

    # ── DISTANCE EXTRACTION ────────────────────────────
    {
        "category": "distance",
        "text":     "hostels within 2km of campus",
        "expected": {"max_distance_km": 2.0}
    },
    {
        "category": "distance",
        "text":     "I need accommodation within 1km",
        "expected": {"max_distance_km": 1.0}
    },
    {
        "category": "distance",
        "text":     "show hostels within 3km of FAST",
        "expected": {"max_distance_km": 3.0}
    },
]

# ─────────────────────────────────────────────────────
# RUN TESTS AND SCORE
# ─────────────────────────────────────────────────────
def check_entity(extracted, expected, category):
    if category == "amenity":
        exp_set = set(expected.get("amenities", []))
        got_set = set(extracted.get("amenities", []))
        # Pass if all expected amenities were found
        return exp_set.issubset(got_set)
    elif category == "budget":
        return extracted.get("budget") == expected.get("budget")
    elif category == "hostel_name":
        return extracted.get("hostel_name") == \
               expected.get("hostel_name")
    elif category == "room_type":
        return extracted.get("room_type") == \
               expected.get("room_type")
    elif category == "distance":
        return extracted.get("max_distance_km") == \
               expected.get("max_distance_km")
    return False

# Run all tests
print("=" * 65)
print("STAYBUDDY — ENTITY EXTRACTION ACCURACY TEST")
print("=" * 65)

results_by_category = {}
all_pass = 0
all_total = 0

for tc in TEST_CASES:
    cat      = tc["category"]
    text     = tc["text"]
    expected = tc["expected"]
    extracted = extract_entities(text)
    passed   = check_entity(extracted, expected, cat)

    if cat not in results_by_category:
        results_by_category[cat] = {"pass": 0, "total": 0,
                                     "failures": []}
    results_by_category[cat]["total"] += 1
    all_total += 1

    if passed:
        results_by_category[cat]["pass"] += 1
        all_pass += 1
    else:
        results_by_category[cat]["failures"].append({
            "text":     text,
            "expected": expected,
            "got":      extracted
        })

# Print results by category
for cat, res in results_by_category.items():
    pct = (res["pass"] / res["total"]) * 100
    status = "✅" if pct >= 80 else "❌"
    print(f"\n{status} {cat.upper()} EXTRACTION")
    print(f"   Accuracy: {res['pass']}/{res['total']} "
          f"= {pct:.0f}%")
    if res["failures"]:
        print("   Failed cases:")
        for f in res["failures"]:
            print(f"     Text    : {f['text']}")
            print(f"     Expected: {f['expected']}")
            print(f"     Got     : {f['got']}")

# Overall score
overall = (all_pass / all_total) * 100
print("\n" + "=" * 65)
print(f"OVERALL ENTITY EXTRACTION ACCURACY: "
      f"{all_pass}/{all_total} = {overall:.1f}%")
print("=" * 65)

# Summary table
print("\nSUMMARY TABLE:")
print(f"{'Category':<15} {'Correct':<10} {'Total':<10} "
      f"{'Accuracy':<10}")
print("-" * 45)
for cat, res in results_by_category.items():
    pct = (res["pass"] / res["total"]) * 100
    print(f"{cat:<15} {res['pass']:<10} {res['total']:<10} "
          f"{pct:.0f}%")
print("-" * 45)
print(f"{'TOTAL':<15} {all_pass:<10} {all_total:<10} "
      f"{overall:.1f}%")