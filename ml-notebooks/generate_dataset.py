"""
StayBuddy - Synthetic Dataset Generator
=============================================
Generates: hostels.csv, students.csv, interactions.csv

Covers:
  - Prelim Implementation Guide requirements
  - All student, parent, warden, admin use cases
  - GPS-based search (UC-STU-001)
  - University-based search (UC-STU-002)
  - Booking flow (room types, food preference)
  - Parent portal (fee breakdown, meal type, electricity)
  - Collaborative filtering (timestamps, session IDs)
  - Content-based filtering (all preference-feature pairs)

Author : Eraj Zaman (22I-1296)
Project: StayBuddy - Intelligent Hostel Management System
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

np.random.seed(42)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
N_HOSTELS_GIRLS = 38
N_HOSTELS_BOYS  = 37
N_STUDENTS      = 200

# ─────────────────────────────────────────────────────────────
# REAL ISLAMABAD COORDINATES (near FAST NUCES H-11)
# FAST NUCES Islamabad campus: 33.6461 N, 72.9928 E
# ─────────────────────────────────────────────────────────────
FAST_LAT = 33.6461
FAST_LNG = 72.9928

islamabad_areas = {
    "H-11"          : {"lat": 33.6461, "lng": 72.9928, "base_dist": 0.5},
    "H-13"          : {"lat": 33.6300, "lng": 73.0100, "base_dist": 1.2},
    "G-11"          : {"lat": 33.6600, "lng": 72.9800, "base_dist": 1.8},
    "G-13"          : {"lat": 33.6550, "lng": 73.0200, "base_dist": 2.5},
    "F-10"          : {"lat": 33.7000, "lng": 72.9700, "base_dist": 3.0},
    "F-11"          : {"lat": 33.7100, "lng": 72.9600, "base_dist": 3.5},
    "I-8"           : {"lat": 33.6700, "lng": 73.0600, "base_dist": 2.8},
    "I-10"          : {"lat": 33.6500, "lng": 73.0500, "base_dist": 3.2},
    "E-11"          : {"lat": 33.7300, "lng": 72.9500, "base_dist": 4.5},
    "Bahria Town"   : {"lat": 33.5300, "lng": 72.9200, "base_dist": 6.0},
    "DHA Phase 2"   : {"lat": 33.5500, "lng": 72.9400, "base_dist": 5.5},
    "Satellite Town": {"lat": 33.6100, "lng": 73.0400, "base_dist": 4.0},
    "Gulberg"       : {"lat": 33.6200, "lng": 73.0800, "base_dist": 5.0},
    "PWD"           : {"lat": 33.6000, "lng": 73.0700, "base_dist": 4.8},
    "Korang Town"   : {"lat": 33.6350, "lng": 73.0900, "base_dist": 3.8},
}

amenities_pool = [
    "WiFi", "Laundry", "Gym", "Study Room", "Cafeteria",
    "CCTV", "Generator", "Parking", "AC", "Hot Water",
    "Library", "Common Room", "Prayer Room", "Security Guard",
    "Water Cooler", "Iron", "Refrigerator", "Microwave",
    "Lounge", "Rooftop Access"
]

hostel_names_girls = [
    "Fatima Girls Hostel",   "Khadija Residence",      "Zainab Girls Inn",
    "Maryam Hostel",         "Aisha Girls Lodge",      "Noor Girls Hostel",
    "Hira Girls Residency",  "Sana Girls Hostel",      "Rabia Girls Inn",
    "Bushra Girls Hostel",   "Sidra Girls Lodge",      "Amna Residence",
    "Umm-e-Hani Hostel",     "Ruqayya Girls Inn",      "Tahira Hostel",
    "Nimra Girls Lodge",     "Safia Girls Residency",  "Hafsa Inn",
    "Sumayya Girls Hostel",  "Asma Girls Lodge",       "Lubna Residence",
    "Razia Girls Hostel",    "Zubaida Inn",            "Salma Girls Lodge",
    "Nadia Girls Residency", "Fozia Girls Hostel",     "Yasmeen Inn",
    "Shabnam Girls Lodge",   "Gulnaz Residence",       "Parveen Girls Hostel",
    "Shazia Girls Inn",      "Muskan Residency",       "Rukhsana Hostel",
    "Fariha Girls Lodge",    "Mehwish Inn",            "Sawera Girls Hostel",
    "Samina Residency",      "Uzma Girls Lodge",
]

hostel_names_boys = [
    "Ali Boys Hostel",       "Umar Residence",         "Usman Boys Inn",
    "Hassan Boys Lodge",     "Hussain Hostel",         "Talha Boys Hostel",
    "Bilal Residency",       "Saad Boys Inn",          "Hamza Hostel",
    "Khalid Boys Lodge",     "Anas Residence",         "Zaid Boys Hostel",
    "Muaaz Inn",             "Talhah Boys Lodge",      "Sufyan Hostel",
    "Wahab Boys Residency",  "Farrukh Inn",            "Kamran Boys Lodge",
    "Imran Hostel",          "Adnan Boys Residency",   "Rizwan Inn",
    "Tariq Boys Hostel",     "Naveed Residency",       "Asif Boys Lodge",
    "Junaid Inn",            "Shahid Boys Hostel",     "Waqar Residency",
    "Faisal Boys Lodge",     "Nasir Inn",              "Sajid Boys Hostel",
    "Umer Residency",        "Ahsan Boys Lodge",       "Zubair Inn",
    "Shoaib Boys Hostel",    "Atif Residency",         "Babar Boys Lodge",
    "Yasir Inn",
]

street_names = [
    "Street 1", "Street 4", "Street 7", "Street 12", "Street 15",
    "Street 20", "Main Boulevard", "Park Road", "College Road",
    "University Road", "Commercial Avenue", "Sector Road",
]


# ─────────────────────────────────────────────────────────────
# 1. HOSTEL DATASET
# ─────────────────────────────────────────────────────────────
def generate_hostel(i, name, h_type):
    area_name   = np.random.choice(list(islamabad_areas.keys()))
    area        = islamabad_areas[area_name]

    lat         = round(area["lat"] + np.random.uniform(-0.005, 0.005), 6)
    lng         = round(area["lng"] + np.random.uniform(-0.005, 0.005), 6)
    distance_km = round(area["base_dist"] + np.random.uniform(-0.3, 0.5), 2)
    distance_km = max(0.3, distance_km)

    street      = np.random.choice(street_names)
    house_no    = np.random.randint(1, 200)
    address     = f"House {house_no}, {street}, {area_name}, Islamabad"

    price_tier  = np.random.choice(
        ["Budget", "Mid-Range", "Premium"], p=[0.40, 0.40, 0.20]
    )

    if price_tier == "Budget":
        single_price     = np.random.randint(8000,  13000)
        double_price     = np.random.randint(6000,  9000)
        dorm_price       = np.random.randint(4000,  7000)
        quality_score    = round(np.random.uniform(2.5, 3.5), 1)
        internet_mbps    = int(np.random.choice([5, 10, 15]))
        electricity_bill = np.random.randint(200, 600)
    elif price_tier == "Mid-Range":
        single_price     = np.random.randint(13000, 22000)
        double_price     = np.random.randint(9000,  15000)
        dorm_price       = np.random.randint(7000,  11000)
        quality_score    = round(np.random.uniform(3.0, 4.2), 1)
        internet_mbps    = int(np.random.choice([15, 25, 50]))
        electricity_bill = np.random.randint(400, 900)
    else:
        single_price     = np.random.randint(22000, 40000)
        double_price     = np.random.randint(15000, 25000)
        dorm_price       = np.random.randint(11000, 18000)
        quality_score    = round(np.random.uniform(4.0, 5.0), 1)
        internet_mbps    = int(np.random.choice([50, 100, 200]))
        electricity_bill = 0

    electricity_included = int(price_tier == "Premium" or np.random.random() < 0.30)
    if electricity_included:
        electricity_bill = 0

    min_am = {"Budget": 4, "Mid-Range": 7, "Premium": 10}[price_tier]
    max_am = {"Budget": 8, "Mid-Range": 12, "Premium": 16}[price_tier]
    chosen = np.random.choice(
        amenities_pool,
        np.random.randint(min_am, max_am + 1),
        replace=False
    ).tolist()

    meal_included = int(np.random.random() < 0.45)
    if meal_included or "Cafeteria" in chosen:
        food_type   = np.random.choice(["Veg", "Non-Veg", "Both"], p=[0.20, 0.40, 0.40])
        food_rating = round(np.random.uniform(2.0, 5.0), 1)
    else:
        food_type   = "None"
        food_rating = None

    room_types = ["Single", "Double"]
    if np.random.random() < 0.60:
        room_types.append("Dormitory")
    if np.random.random() < 0.30:
        room_types.append("Triple")

    cleanliness = round(np.random.uniform(2.5, 5.0), 1)
    security    = round(np.random.uniform(2.5, 5.0), 1)
    management  = round(np.random.uniform(2.5, 5.0), 1)
    overall     = round(np.mean([quality_score, cleanliness, security, management]), 1)

    study_env = round(
        (int("Study Room" in chosen) * 0.40 +
         int("Library"    in chosen) * 0.30 +
         int("WiFi"       in chosen) * 0.20 +
         (1 - min(distance_km / 8, 1)) * 0.10), 2
    )

    noise_level = int(np.random.randint(1, 6))
    curfew_hour = int(np.random.choice([20, 21, 22, 23]) if h_type == "Girls"
                      else np.random.choice([22, 23, 0]))
    total_rooms = int(np.random.randint(10, 50))
    warden_phone = f"03{np.random.randint(10,50)}-{np.random.randint(1000000,9999999)}"

    return {
        # Identifiers
        "hostel_id"               : f"HST-{i+1:03d}",
        "hostel_name"             : name,
        "hostel_type"             : h_type,
        "verified"                : int(np.random.random() < 0.80),
        "year_established"        : int(np.random.randint(2005, 2024)),
        # Location
        "area"                    : area_name,
        "city"                    : "Islamabad",
        "hostel_address"          : address,
        "latitude"                : lat,
        "longitude"               : lng,
        "distance_from_fast_km"   : distance_km,
        "transport_nearby"        : int(np.random.random() < 0.60),
        # Pricing
        "price_tier"              : price_tier,
        "single_room_price"       : int(single_price),
        "double_room_price"       : int(double_price),
        "dorm_room_price"         : int(dorm_price),
        "electricity_included"    : electricity_included,
        "electricity_bill_est"    : int(electricity_bill),
        # Food
        "meal_included"           : meal_included,
        "food_type"               : food_type,
        "food_rating"             : food_rating,
        # Rooms
        "room_types_available"    : json.dumps(room_types),
        "total_rooms"             : total_rooms,
        "capacity"                : int(total_rooms * np.random.randint(2, 5)),
        "available_rooms"         : int(np.random.randint(0, min(20, total_rooms))),
        # Amenities
        "amenities"               : json.dumps(chosen),
        "has_wifi"                : int("WiFi"            in chosen),
        "has_gym"                 : int("Gym"             in chosen),
        "has_study_room"          : int("Study Room"      in chosen),
        "has_cafeteria"           : int("Cafeteria"       in chosen),
        "has_laundry"             : int("Laundry"         in chosen),
        "has_ac"                  : int("AC"              in chosen),
        "has_generator"           : int("Generator"       in chosen),
        "has_security_guard"      : int("Security Guard"  in chosen),
        "has_cctv"                : int("CCTV"            in chosen),
        "has_hot_water"           : int("Hot Water"       in chosen),
        "has_library"             : int("Library"         in chosen),
        "has_parking"             : int("Parking"         in chosen),
        "has_prayer_room"         : int("Prayer Room"     in chosen),
        "has_common_room"         : int("Common Room"     in chosen),
        # Quality
        "internet_speed_mbps"     : internet_mbps,
        "noise_level"             : noise_level,
        "curfew_hour"             : curfew_hour,
        "study_environment_score" : study_env,
        # Ratings
        "overall_rating"          : overall,
        "cleanliness_rating"      : cleanliness,
        "security_rating"         : security,
        "management_rating"       : management,
        "warden_responsiveness"   : round(np.random.uniform(1.0, 5.0), 1),
        "total_reviews"           : int(np.random.randint(5, 150)),
        # Contact
        "warden_contact_phone"    : warden_phone,
    }


print("Generating hostels...")
hostels = []
idx = 0
for name in hostel_names_girls:
    hostels.append(generate_hostel(idx, name, "Girls"))
    idx += 1
for name in hostel_names_boys:
    hostels.append(generate_hostel(idx, name, "Boys"))
    idx += 1

hostels_df = pd.DataFrame(hostels)
print(f"✅ Hostels generated  : {len(hostels_df)}")
print(f"   Girls             : {len(hostels_df[hostels_df.hostel_type=='Girls'])}")
print(f"   Boys              : {len(hostels_df[hostels_df.hostel_type=='Boys'])}")
print(f"   Total features    : {len(hostels_df.columns)}")
print(hostels_df[[
    "hostel_name","hostel_type","price_tier","single_room_price",
    "distance_from_fast_km","food_type","overall_rating"
]].head(5).to_string())


# ─────────────────────────────────────────────────────────────
# 2. STUDENT PROFILES
# ─────────────────────────────────────────────────────────────
departments = [
    "Computer Science", "Electrical Engineering", "BBA",
    "Social Sciences",  "Civil Engineering",      "Software Engineering",
    "Cyber Security",   "Data Science"
]

university_coords = [
    (33.6461, 72.9928),  # FAST H-11
    (33.7200, 73.0479),  # NUST
    (33.7294, 73.0931),  # COMSATS
    (33.6844, 73.0479),  # QAU
]

print("\nGenerating students...")
students = []
for i in range(N_STUDENTS):
    gender       = np.random.choice(["Male", "Female"], p=[0.55, 0.45])
    budget_min   = int(np.random.randint(5000,  15000))
    budget_max   = int(budget_min + np.random.randint(5000, 25000))
    max_dist     = round(np.random.uniform(0.5, 7.0), 1)
    study_pref   = round(np.random.uniform(0, 1), 2)
    food_pref    = np.random.choice(["Veg", "Non-Veg", "Both"], p=[0.20, 0.45, 0.35])
    room_pref    = np.random.choice(["Single", "Double", "Dormitory"], p=[0.35, 0.40, 0.25])
    uni_coord    = university_coords[np.random.randint(0, len(university_coords))]
    student_lat  = round(uni_coord[0] + np.random.uniform(-0.01, 0.01), 6)
    student_lng  = round(uni_coord[1] + np.random.uniform(-0.01, 0.01), 6)
    must_have    = np.random.choice(
        amenities_pool, np.random.randint(1, 5), replace=False
    ).tolist()

    students.append({
        # Identifiers
        "student_id"            : f"STU-{i+1:03d}",
        "gender"                : gender,
        "department"            : np.random.choice(departments),
        "year"                  : int(np.random.choice([1, 2, 3, 4])),
        # GPS
        "latitude"              : student_lat,
        "longitude"             : student_lng,
        # Budget
        "budget_min"            : budget_min,
        "budget_max"            : budget_max,
        "price_sensitivity"     : round(np.random.uniform(0, 1), 2),
        # Location
        "max_distance_km"       : max_dist,
        "needs_transport"       : int(max_dist > 3.5),
        # Type & Room & Food
        "preferred_type"        : "Girls" if gender == "Female" else "Boys",
        "preferred_room_type"   : room_pref,
        "food_preference"       : food_pref,
        # Lifestyle
        "study_preference"      : study_pref,
        "social_preference"     : round(1 - study_pref, 2),
        "comfort_preference"    : round(np.random.uniform(0, 1), 2),
        "noise_tolerance"       : round(np.random.uniform(0, 1), 2),
        "curfew_flexibility"    : round(np.random.uniform(0, 1), 2),
        # Amenity priorities
        "must_have_amenities"   : json.dumps(must_have),
        "priority_wifi"         : int("WiFi"          in must_have),
        "priority_gym"          : int("Gym"           in must_have),
        "priority_study_room"   : int("Study Room"    in must_have),
        "priority_cafeteria"    : int("Cafeteria"     in must_have),
        "priority_laundry"      : int("Laundry"       in must_have),
        "priority_ac"           : int("AC"            in must_have),
        "priority_hot_water"    : int("Hot Water"     in must_have),
        "priority_generator"    : int("Generator"     in must_have),
    })

students_df = pd.DataFrame(students)
print(f"✅ Students generated : {len(students_df)}")
print(f"   Female            : {len(students_df[students_df.gender=='Female'])}")
print(f"   Male              : {len(students_df[students_df.gender=='Male'])}")
print(f"   Total features    : {len(students_df.columns)}")
print(students_df[[
    "student_id","gender","budget_min","budget_max",
    "food_preference","preferred_room_type","study_preference"
]].head(5).to_string())


# ─────────────────────────────────────────────────────────────
# 3. INTERACTION DATA
# Timestamps: Sept 2025 → Feb 2026 (one full semester)
# Session IDs: group related browsing behaviour
# Weights: view=1, save=3, booking_attempt=4, booking=5
# ─────────────────────────────────────────────────────────────
print("\nGenerating interactions...")

BASE_DATE  = datetime(2025, 9, 1)
END_DATE   = datetime(2026, 2, 27)
DATE_RANGE = (END_DATE - BASE_DATE).days

interactions = []
iid = 1

for _, student in students_df.iterrows():
    sid       = student["student_id"]
    s_type    = student["preferred_type"]

    gender_pool = hostels_df[hostels_df["hostel_type"] == s_type]
    budget_pool = gender_pool[
        gender_pool["single_room_price"] <= student["budget_max"] * 1.3
    ]
    if len(budget_pool) < 5:
        budget_pool = gender_pool

    n_sessions = np.random.randint(1, 4)

    for sess_num in range(n_sessions):
        session_id   = f"SESS-{sid}-{sess_num+1:02d}"
        session_date = BASE_DATE + timedelta(days=int(np.random.randint(0, DATE_RANGE)))

        n_views = np.random.randint(3, 12)
        viewed  = budget_pool.sample(
            min(n_views, len(budget_pool)), replace=False
        )["hostel_id"].tolist()

        for j, hid in enumerate(viewed):
            ts = session_date + timedelta(minutes=j * int(np.random.randint(2, 10)))
            interactions.append({
                "interaction_id"  : f"INT-{iid:05d}",
                "student_id"      : sid,
                "hostel_id"       : hid,
                "interaction_type": "view",
                "weight"          : 1,
                "rating"          : None,
                "session_id"      : session_id,
                "timestamp"       : ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
            iid += 1

        n_saves = np.random.randint(1, min(4, len(viewed) + 1))
        saved   = np.random.choice(viewed, n_saves, replace=False).tolist()
        for hid in saved:
            ts = session_date + timedelta(hours=1, minutes=int(np.random.randint(0, 30)))
            interactions.append({
                "interaction_id"  : f"INT-{iid:05d}",
                "student_id"      : sid,
                "hostel_id"       : hid,
                "interaction_type": "save",
                "weight"          : 3,
                "rating"          : None,
                "session_id"      : session_id,
                "timestamp"       : ts.strftime("%Y-%m-%d %H:%M:%S"),
            })
            iid += 1

    all_saves = [
        r["hostel_id"] for r in interactions
        if r["student_id"] == sid and r["interaction_type"] == "save"
    ]
    if not all_saves:
        continue

    if np.random.random() < 0.40:
        attempt_ts = END_DATE - timedelta(days=int(np.random.randint(1, 30)))
        interactions.append({
            "interaction_id"  : f"INT-{iid:05d}",
            "student_id"      : sid,
            "hostel_id"       : np.random.choice(all_saves),
            "interaction_type": "booking_attempt",
            "weight"          : 4,
            "rating"          : None,
            "session_id"      : f"SESS-{sid}-BOOK",
            "timestamp"       : attempt_ts.strftime("%Y-%m-%d %H:%M:%S"),
        })
        iid += 1

    if np.random.random() < 0.60:
        booking_ts = END_DATE - timedelta(days=int(np.random.randint(1, 20)))
        interactions.append({
            "interaction_id"  : f"INT-{iid:05d}",
            "student_id"      : sid,
            "hostel_id"       : np.random.choice(all_saves),
            "interaction_type": "booking",
            "weight"          : 5,
            "rating"          : round(np.random.uniform(2.5, 5.0), 1),
            "session_id"      : f"SESS-{sid}-BOOK",
            "timestamp"       : booking_ts.strftime("%Y-%m-%d %H:%M:%S"),
        })
        iid += 1

interactions_df = pd.DataFrame(interactions)
print(f"✅ Interactions generated : {len(interactions_df)}")
print(interactions_df["interaction_type"].value_counts().to_string())
print(f"\n   Students with bookings : "
      f"{interactions_df[interactions_df.interaction_type=='booking']['student_id'].nunique()}")
print(f"   Hostels rated          : "
      f"{interactions_df[interactions_df.rating.notna()]['hostel_id'].nunique()}")
print(f"   Date range             : "
      f"{interactions_df.timestamp.min()} → {interactions_df.timestamp.max()}")


# ─────────────────────────────────────────────────────────────
# 4. SAVE ALL FILES
# ─────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
hostels_df.to_csv("data/hostels.csv",           index=False)
students_df.to_csv("data/students.csv",          index=False)
interactions_df.to_csv("data/interactions.csv",  index=False)

print("\n" + "="*60)
print("✅ ALL FILES SAVED TO data/")
print(f"   hostels.csv      : {len(hostels_df):>5} rows | {len(hostels_df.columns):>2} features")
print(f"   students.csv     : {len(students_df):>5} rows | {len(students_df.columns):>2} features")
print(f"   interactions.csv : {len(interactions_df):>5} rows |  8 features")
print()
print(" USE CASE COVERAGE CONFIRMED:")
print("   ✓ latitude/longitude on hostels   — UC-STU-001 GPS search")
print("   ✓ latitude/longitude on students  — UC-STU-001 GPS search")
print("   ✓ hostel_address                  — Zarnab DB schema match")
print("   ✓ food_type on hostels            — Parent UC-P1 fee breakdown")
print("   ✓ food_preference on students     — Veg/Non-Veg/Both matching")
print("   ✓ room_types_available            — Booking use case")
print("   ✓ preferred_room_type on students — Booking preference")
print("   ✓ electricity_included/bill       — Parent UC-P1 cost breakdown")
print("   ✓ warden_contact_phone            — Parent contact warden")
print("   ✓ timestamp on interactions       — Time-decay collab filtering")
print("   ✓ session_id on interactions      — Session behaviour patterns")
print("   ✓ interaction weights 1/3/4/5     — Collab filtering signals")
print("   ✓ Girls/Boys only (no Mixed)      — Cultural accuracy")
print("="*60)
print(" Dataset complete — ready for ML model building!")