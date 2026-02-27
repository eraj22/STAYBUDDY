"""
╔══════════════════════════════════════════════════════════════════╗
║         StayBuddy — Intelligent Hostel Discovery Platform        ║
║         Streamlit Demo Interface — Prelim Presentation           ║
╠══════════════════════════════════════════════════════════════════╣
║  Author  : Eraj Zaman (22I-1296)                                 ║
║  Project : StayBuddy - Intelligent Hostel Management System      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "StayBuddy — Intelligent Hostel Finder",
    page_icon   = "🏠",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ──────────────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #7f8c8d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .hostel-card {
        background: #f8f9fa;
        border-left: 4px solid #2ecc71;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.8rem 0;
    }
    .score-badge {
        background: #2ecc71;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .why-box {
        background: #eaf4fb;
        border: 1px solid #3498db;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #2c3e50;
        margin-top: 0.5rem;
    }
    .intel-badge {
        background: #e8f5e9;
        border: 1px solid #2ecc71;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #1a5c35;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 0.8rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")


# ──────────────────────────────────────────────────────────────────
# LOAD DATA (cached so it only loads once)
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    hostels_df      = pd.read_csv(os.path.join(DATA_DIR, "hostels.csv"))
    students_df     = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
    interactions_df = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
    interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
    return hostels_df, students_df, interactions_df

@st.cache_resource
def load_models():
    hostel_matrix    = np.load(os.path.join(MODEL_DIR, "hostel_feature_matrix.npy"))
    predicted_matrix = pd.read_csv(os.path.join(MODEL_DIR, "predicted_matrix.csv"), index_col=0)
    interaction_matrix = pd.read_csv(os.path.join(MODEL_DIR, "interaction_matrix.csv"), index_col=0)
    cold_start_models  = joblib.load(os.path.join(MODEL_DIR, "cold_start_models.pkl"))
    with open(os.path.join(MODEL_DIR, "hybrid_config.json")) as f:
        hybrid_config = json.load(f)
    with open(os.path.join(MODEL_DIR, "hybrid_metrics.json")) as f:
        hybrid_metrics = json.load(f)
    with open(os.path.join(MODEL_DIR, "cf_metrics.json")) as f:
        cf_metrics = json.load(f)
    return (hostel_matrix, predicted_matrix, interaction_matrix,
            cold_start_models, hybrid_config, hybrid_metrics, cf_metrics)

hostels_df, students_df, interactions_df = load_data()
(hostel_matrix, predicted_matrix, interaction_matrix,
 cold_start_models, hybrid_config, hybrid_metrics, cf_metrics) = load_models()

best_alpha  = hybrid_config["best_alpha"]
type_alphas = hybrid_config["type_alphas"]

TECH_DEPTS = [
    "Computer Science","Electrical Engineering",
    "Software Engineering","Cyber Security","Data Science"
]
DEMO_FEATURES = [
    "budget_max","max_distance_km","study_preference",
    "price_sensitivity","comfort_preference","noise_tolerance",
    "priority_wifi","priority_study_room","priority_ac",
    "priority_generator","needs_transport"
]


# ──────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────
def classify_student_type(study_pref, price_sens, comfort_pref):
    if study_pref > 0.70:   return "study_focused"
    if price_sens > 0.70:   return "budget_conscious"
    if comfort_pref > 0.70: return "comfort_seeking"
    return "balanced"

def food_compat(student_pref, hostel_food):
    if hostel_food == "None":      return 0.8
    if student_pref == "Both":     return 1.0
    if student_pref == hostel_food: return 1.0
    if hostel_food == "Both":      return 0.9
    return 0.3

def room_compat(preferred, available_json):
    try:
        available = json.loads(available_json)
    except:
        return 0.5
    if preferred in available: return 1.0
    alts = {"Single":["Double"],"Double":["Single","Triple"],
            "Dormitory":["Triple"],"Triple":["Double","Dormitory"]}
    return 0.6 if any(a in available for a in alts.get(preferred,[])) else 0.3

def build_ad_hoc_student_vector(gender, department, budget_max,
                                 max_dist, study_pref, food_pref,
                                 room_type, price_sens, comfort_pref,
                                 noise_tol, curfew_flex, needs_transport,
                                 must_have):
    """Build a student vector from form inputs — no student_id needed."""
    price_score   = np.clip(1-(budget_max/hostels_df["single_room_price"].max()),0,1)
    dist_score    = np.clip(1-(max_dist/hostels_df["distance_from_fast_km"].max()),0,1)
    safety_score  = 0.90 if gender=="Female" else 0.70
    food_map      = {"Veg":0.33,"Non-Veg":0.66,"Both":1.0}
    food_score    = food_map.get(food_pref, 0.5)
    internet_score= 0.90 if department in TECH_DEPTS else 0.50
    amenity_score = min(len(must_have)/14, 1.0)

    base = np.array([
        price_score, dist_score, study_pref, study_pref,
        safety_score, comfort_pref, amenity_score, price_sens,
        internet_score, 1-noise_tol, food_score, curfew_flex,
        float(needs_transport), 1.0 if food_pref!="None" else 0.0,
        price_sens*0.50
    ])
    weights = np.array([
        price_sens, 1.0, study_pref, study_pref,
        0.90 if gender=="Female" else 0.70,
        comfort_pref, min(len(must_have)/5,1.0), price_sens,
        internet_score, 1-noise_tol, 0.80,
        curfew_flex, float(needs_transport), 0.70, price_sens*0.50
    ])
    weights = weights/(weights.sum()+1e-9)
    return base*weights

def get_ad_hoc_recommendations(
    gender, department, budget_max, max_dist, study_pref,
    food_pref, room_type, price_sens, comfort_pref, noise_tol,
    curfew_flex, needs_transport, must_have, top_k=5
):
    """
    Core recommendation function for ad-hoc (new) students.
    Uses hybrid CB+CF logic without requiring a student_id.
    """
    hostel_type = "Girls" if gender=="Female" else "Boys"
    gender_mask = hostels_df["hostel_type"]==hostel_type
    filt_h      = hostels_df[gender_mask].copy()
    filt_m      = hostel_matrix[gender_mask.values]

    # ── Content-Based Score ───────────────────────────────────────
    svec  = build_ad_hoc_student_vector(
        gender, department, budget_max, max_dist, study_pref,
        food_pref, room_type, price_sens, comfort_pref,
        noise_tol, curfew_flex, needs_transport, must_have
    ).reshape(1,-1)
    sims  = cosine_similarity(svec, filt_m)[0]
    sims *= filt_h["food_type"].apply(
        lambda ft: food_compat(food_pref, ft)
    ).values
    sims *= (0.7+0.3*filt_h["room_types_available"].apply(
        lambda rt: room_compat(room_type, rt)
    ).values)
    sims *= (0.85+0.15*(filt_h["available_rooms"]>0).astype(float).values)
    cb_scores = pd.Series(sims, index=filt_h["hostel_id"].values)
    if cb_scores.max()>cb_scores.min():
        cb_scores = (cb_scores-cb_scores.min())/(cb_scores.max()-cb_scores.min())

    # ── CF Score: use demographic clustering (cold-start) ─────────
    model = cold_start_models.get(gender)
    if model is not None:
        feat_vec = np.array([[
            budget_max, max_dist, study_pref, price_sens,
            comfort_pref, noise_tol,
            1 if "WiFi" in must_have else 0,
            1 if "Study Room" in must_have else 0,
            1 if "AC" in must_have else 0,
            1 if "Generator" in must_have else 0,
            int(needs_transport)
        ]])
        scaled  = model["scaler"].transform(feat_vec)
        cluster = model["kmeans"].predict(scaled)[0]
        labels  = model["kmeans"].labels_
        sim_idx = [
            model["indices"][i]
            for i,lbl in enumerate(labels) if lbl==cluster
        ]
        sim_sids = students_df.loc[sim_idx,"student_id"].tolist()
        avg = predicted_matrix.loc[
            predicted_matrix.index.isin(sim_sids)
        ].mean(axis=0)
        cf_scores = avg[avg.index.isin(filt_h["hostel_id"].values)]
    else:
        cf_scores = filt_h.set_index("hostel_id")["overall_rating"]/5

    if cf_scores.max()>cf_scores.min():
        cf_scores = (cf_scores-cf_scores.min())/(cf_scores.max()-cf_scores.min())

    # ── Hybrid fusion ─────────────────────────────────────────────
    stype = classify_student_type(study_pref, price_sens, comfort_pref)
    alpha = type_alphas.get(stype, best_alpha)
    idx   = cb_scores.index.union(cf_scores.index)
    hybrid = (
        alpha * cb_scores.reindex(idx, fill_value=0) +
        (1-alpha) * cf_scores.reindex(idx, fill_value=0)
    )
    if hybrid.max()>hybrid.min():
        hybrid = (hybrid-hybrid.min())/(hybrid.max()-hybrid.min())

    top_ids = hybrid.nlargest(top_k).index.tolist()
    results = filt_h[filt_h["hostel_id"].isin(top_ids)].copy()
    results["hybrid_score"] = results["hostel_id"].map(hybrid)
    results["cb_score"]     = results["hostel_id"].map(
        cb_scores.reindex(results["hostel_id"], fill_value=0)
    )
    results["cf_score"]     = results["hostel_id"].map(
        cf_scores.reindex(results["hostel_id"], fill_value=0)
    )
    results["alpha_used"]   = alpha
    results["student_type"] = stype

    return results.sort_values("hybrid_score",ascending=False).reset_index(drop=True)

def generate_explanation(row, gender, food_pref, room_type,
                          budget_max, max_dist, study_pref,
                          department, alpha):
    reasons = []
    if row["single_room_price"] <= budget_max:
        pct = (row["single_room_price"]/budget_max)*100
        reasons.append(f"✅ Within budget (PKR {row['single_room_price']:,} = {pct:.0f}% of your max)")
    if row["distance_from_fast_km"] <= max_dist:
        reasons.append(f"📍 Close to FAST ({row['distance_from_fast_km']}km)")
    if study_pref>0.6 and row["study_environment_score"]>0.5:
        reasons.append(f"📚 Strong study environment ({row['study_environment_score']})")
    if department in TECH_DEPTS and row["has_wifi"]:
        reasons.append(f"🌐 Fast WiFi ({row['internet_speed_mbps']} Mbps)")
    if gender=="Female" and row["security_rating"]>=4.0:
        reasons.append(f"🔒 High security ({row['security_rating']}/5)")
    if food_pref==row["food_type"]:
        reasons.append(f"🍽️ Perfect food match ({row['food_type']})")
    elif row["food_type"]=="Both":
        reasons.append(f"🍽️ Flexible food (Veg & Non-Veg available)")
    try:
        rooms = json.loads(row["room_types_available"])
        if room_type in rooms:
            reasons.append(f"🛏️ {room_type} room available")
    except:
        pass
    if row["overall_rating"]>=4.0:
        reasons.append(f"⭐ Highly rated ({row['overall_rating']}/5 from {row['total_reviews']} reviews)")
    if row["cf_score"]>0.5:
        reasons.append(f"👥 Popular with similar students")
    return reasons[:4]

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1,lon1,lat2,lon2 = map(np.radians,[lat1,lon1,lat2,lon2])
    dlat,dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arcsin(np.sqrt(a))


# ──────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 StayBuddy")
    st.markdown("*Intelligent Hostel Finder*")
    st.divider()

    page = st.radio(
        "Navigate",
        ["🏠 Home",
         "🔍 Find My Hostel",
         "📍 GPS Search",
         "📊 Model Performance"],
        label_visibility="hidden"
    )

    st.divider()
    st.markdown("**System Info**")
    st.markdown(f"🏨 Hostels: **{len(hostels_df)}**")
    st.markdown(f"👤 Students: **{len(students_df)}**")
    st.markdown(f"🔄 Interactions: **{len(interactions_df):,}**")
    st.markdown(f"⚙️ Optimal α: **{best_alpha}**")
    st.divider()
    st.caption("Eraj Zaman — 22I-1296")
    st.caption("FAST NUCES Islamabad")


# ══════════════════════════════════════════════════════════════════
# PAGE 1: HOME
# ══════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<div class="main-header">🏠 StayBuddy</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Intelligent Hostel Discovery for FAST NUCES Students</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Hostels", len(hostels_df), "Islamabad")
    with col2:
        st.metric("Students", len(students_df), "Profiles")
    with col3:
        st.metric("Interactions", f"{len(interactions_df):,}", "Logged")
    with col4:
        st.metric("Hybrid MAP", f"{hybrid_metrics['Hybrid']['MAP']:.4f}", "↑ beats CB & CF")

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🧠 Why This Is INTELLIGENT")
        st.markdown("""
| ❌ Automated | ✅ Our System |
|---|---|
| Hard budget filter | 15-dim cosine similarity |
| Fixed rules | SVD discovers hidden patterns |
| One-size-fits-all | Adaptive weights per student |
| No explanation | Every rec explains itself |
| Hardcoded weights | α learned via cross-validation |
| Ignores time | Time-decay on interactions |
| Fails new users | Cold-start clustering |
        """)

    with col_r:
        st.markdown("### 🏗️ Architecture")
        st.code("""
STUDENT INPUT (budget, location, preferences)
             │
    ┌────────┴────────┐
    ▼                 ▼
CONTENT-BASED    COLLABORATIVE
  Cosine Sim      SVD (k=25)
  15 features     Time-decay
    │                 │
    └────────┬────────┘
             ▼
        HYBRID ENGINE
     α × CB + (1-α) × CF
     α = 0.3 (LEARNED)
             │
             ▼
    TOP-5 + EXPLANATIONS
        """, language="")

    st.divider()
    st.markdown("### 📋 Use Cases Covered")
    uc_col1, uc_col2 = st.columns(2)
    with uc_col1:
        st.success("✅ UC-STU-001: GPS-based hostel search")
        st.success("✅ UC-STU-002: University proximity search")
        st.success("✅ Budget / food / room type matching")
        st.success("✅ Gender-appropriate filtering")
    with uc_col2:
        st.success("✅ Parent UC-P1: Safety scoring")
        st.success("✅ Cold start for new students")
        st.success("✅ Admin: 97.33% hostel coverage")
        st.success("✅ Dual CB+CF explainability")


# ══════════════════════════════════════════════════════════════════
# PAGE 2: FIND MY HOSTEL
# ══════════════════════════════════════════════════════════════════
elif page == "🔍 Find My Hostel":
    st.markdown("## 🔍 Find My Hostel")
    st.markdown("*Enter your preferences below. The intelligent hybrid engine will find your best matches.*")

    with st.form("student_form"):
        st.markdown("### 👤 Your Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender     = st.selectbox("Gender", ["Male","Female"])
            department = st.selectbox("Department", [
                "Computer Science","Software Engineering",
                "Electrical Engineering","Cyber Security",
                "Data Science","BBA","Civil Engineering",
                "Social Sciences"
            ])
            year = st.selectbox("Year", [1,2,3,4])

        with col2:
            budget_min = st.number_input("Min Budget (PKR/month)", 5000, 30000, 8000, 1000)
            budget_max = st.number_input("Max Budget (PKR/month)", 8000, 50000, 20000, 1000)
            max_dist   = st.slider("Max Distance from FAST (km)", 0.5, 8.0, 3.0, 0.5)

        with col3:
            food_pref  = st.selectbox("Food Preference", ["Both","Veg","Non-Veg"])
            room_type  = st.selectbox("Preferred Room Type", ["Single","Double","Dormitory"])
            top_k      = st.slider("Number of Recommendations", 3, 10, 5)

        st.divider()
        st.markdown("### ⚙️ Lifestyle Preferences")
        col4, col5, col6 = st.columns(3)
        with col4:
            study_pref  = st.slider("Study Focus", 0.0, 1.0, 0.6, 0.05,
                                     help="0=social, 1=study-focused")
            price_sens  = st.slider("Price Sensitivity", 0.0, 1.0, 0.6, 0.05,
                                     help="0=not sensitive, 1=very cost-conscious")
        with col5:
            comfort_pref= st.slider("Comfort Priority", 0.0, 1.0, 0.5, 0.05)
            noise_tol   = st.slider("Noise Tolerance", 0.0, 1.0, 0.3, 0.05,
                                     help="0=need quiet, 1=tolerant of noise")
        with col6:
            curfew_flex = st.slider("Curfew Flexibility", 0.0, 1.0, 0.5, 0.05,
                                     help="0=need late access, 1=okay with early curfew")
            needs_transport = st.checkbox("Need public transport nearby",
                                          value=(max_dist>3.0))

        st.divider()
        st.markdown("### 🏷️ Must-Have Amenities")
        amenity_options = [
            "WiFi","Study Room","AC","Hot Water","Laundry",
            "Gym","Generator","CCTV","Security Guard",
            "Prayer Room","Cafeteria","Parking"
        ]
        must_have = st.multiselect(
            "Select amenities you require",
            amenity_options,
            default=["WiFi","Hot Water"]
        )

        submitted = st.form_submit_button(
            "🔍 Find My Best Hostels",
            use_container_width=True,
            type="primary"
        )

    if submitted:
        if budget_min >= budget_max:
            st.error("❌ Min budget must be less than max budget")
        else:
            with st.spinner("🧠 Running intelligent hybrid recommendation..."):
                stype = classify_student_type(study_pref, price_sens, comfort_pref)
                alpha = type_alphas.get(stype, best_alpha)

                recs = get_ad_hoc_recommendations(
                    gender, department, budget_max, max_dist,
                    study_pref, food_pref, room_type,
                    price_sens, comfort_pref, noise_tol,
                    curfew_flex, needs_transport, must_have, top_k
                )

            st.success(f"✅ Found {len(recs)} recommendations!")

            # Intelligence summary
            st.markdown(
                f'<div class="intel-badge">'
                f'🧠 Student type: <b>{stype}</b> | '
                f'α = <b>{alpha}</b> (CB={alpha:.0%}, CF={1-alpha:.0%}) — '
                f'learned via cross-validation, not hardcoded'
                f'</div>',
                unsafe_allow_html=True
            )

            st.divider()
            st.markdown(f"### 🏠 Your Top {len(recs)} Hostel Recommendations")

            for rank, (_, row) in enumerate(recs.iterrows(), 1):
                score_pct = row["hybrid_score"]*100
                score_color = (
                    "#2ecc71" if score_pct>=70 else
                    "#f39c12" if score_pct>=50 else "#e74c3c"
                )

                with st.container():
                    c1, c2 = st.columns([3,1])
                    with c1:
                        st.markdown(
                            f"**#{rank} {row['hostel_name']}** "
                            f"[{row['hostel_type']}] — {row['area']}, Islamabad"
                        )
                    with c2:
                        st.markdown(
                            f'<span class="score-badge" '
                            f'style="background:{score_color}">'
                            f'{score_pct:.1f}% Match</span>',
                            unsafe_allow_html=True
                        )

                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Price", f"PKR {row['single_room_price']:,}/mo")
                    col_b.metric("Distance", f"{row['distance_from_fast_km']} km")
                    col_c.metric("Rating", f"{row['overall_rating']}/5 ⭐")
                    col_d.metric("Rooms", f"{row['available_rooms']} avail")

                    # Score breakdown
                    st.markdown(
                        f"CB Score: `{row['cb_score']:.3f}` × α={alpha} | "
                        f"CF Score: `{row['cf_score']:.3f}` × {1-alpha} | "
                        f"**Hybrid: `{row['hybrid_score']:.3f}`**"
                    )

                    # Explanation
                    reasons = generate_explanation(
                        row, gender, food_pref, room_type,
                        budget_max, max_dist, study_pref, department, alpha
                    )
                    if reasons:
                        st.markdown(
                            '<div class="why-box">💡 <b>Why recommended:</b> '
                            + " &nbsp;|&nbsp; ".join(reasons) + '</div>',
                            unsafe_allow_html=True
                        )

                    # Quick extras bar
                    try:
                        rooms_avail = json.loads(row["room_types_available"])
                    except:
                        rooms_avail = []
                    extras = []
                    if row["meal_included"]:    extras.append(f"🍽️ {row['food_type']} meals")
                    if row["electricity_included"]: extras.append("⚡ Electricity incl.")
                    if row["has_wifi"]:         extras.append(f"🌐 WiFi {row['internet_speed_mbps']}Mbps")
                    if row["has_study_room"]:   extras.append("📚 Study room")
                    if row["has_security_guard"]: extras.append("🔒 Security guard")
                    if extras:
                        st.caption("  ·  ".join(extras))

                    # ── UC-STU-005: Full hostel detail expander ───────────
                    with st.expander(f"📋 Full Details — {row['hostel_name']}"):
                        d1, d2, d3 = st.columns(3)

                        with d1:
                            st.markdown("**📍 Location & Access**")
                            st.write(f"Area: **{row['area']}**")
                            st.write(f"Distance from FAST: **{row['distance_from_fast_km']} km**")
                            st.write(f"GPS: `{row['latitude']:.4f}, {row['longitude']:.4f}`")
                            st.write(f"Transport nearby: **{'Yes ✅' if row['transport_nearby'] else 'No ❌'}**")
                            st.write(f"Curfew: **{int(row['curfew_hour']):02d}:00** {'(No curfew)' if row['curfew_hour']==0 else ''}")
                            st.write(f"Total capacity: **{row['total_capacity']} students**")
                            st.write(f"Available rooms: **{row['available_rooms']}**")

                        with d2:
                            st.markdown("**💰 Fee Breakdown (Parent UC-P1)**")
                            base_rent = int(row['single_room_price'])
                            elec_cost = 0 if row['electricity_included'] else 2000
                            meal_cost = 0 if not row['meal_included'] else (
                                4500 if row['food_type'] == 'Both' else 3500
                            )
                            total_est = base_rent + elec_cost + meal_cost

                            st.write(f"Base rent: **PKR {base_rent:,}/mo**")
                            if row['electricity_included']:
                                st.write("Electricity: **Included ✅**")
                            else:
                                st.write(f"Electricity (est.): **PKR {elec_cost:,}/mo**")
                            if row['meal_included']:
                                st.write(f"Meals ({row['food_type']}): **Included ✅**")
                            else:
                                st.write(f"Meals (est.): **PKR {meal_cost:,}/mo**")
                            st.divider()
                            st.write(f"**Estimated Total: PKR {total_est:,}/mo**")
                            st.caption("*Meal/electricity estimates based on market rates")

                            st.markdown("**🛏️ Room Types Available**")
                            try:
                                for rt in json.loads(row["room_types_available"]):
                                    st.write(f"  • {rt}")
                            except:
                                st.write("  • Information unavailable")

                        with d3:
                            st.markdown("**🔒 Safety & Security (Parent UC-P1)**")
                            st.write(f"Security rating: **{row['security_rating']}/5**")
                            st.write(f"CCTV: **{'Yes ✅' if row['has_cctv'] else 'No ❌'}**")
                            st.write(f"Security guard: **{'Yes ✅' if row['has_security_guard'] else 'No ❌'}**")
                            st.write(f"Verified hostel: **{'Yes ✅' if row['verified'] else 'No ❌'}**")
                            st.write(f"Cleanliness: **{row['cleanliness_rating']}/5**")
                            st.write(f"Noise level: **{row['noise_level']}/5**")

                            st.markdown("**📞 Warden Contact**")
                            st.write(f"Phone: `{row['warden_contact_phone']}`")
                            st.write(f"Study env score: **{row['study_environment_score']}/1.0**")
                            st.write(f"Internet speed: **{row['internet_speed_mbps']} Mbps**")
                            st.write(f"Overall rating: **{row['overall_rating']}/5** ({row['total_reviews']} reviews)")

                    st.divider()


# ══════════════════════════════════════════════════════════════════
# PAGE 3: GPS SEARCH
# ══════════════════════════════════════════════════════════════════
elif page == "📍 GPS Search":
    st.markdown("## 📍 GPS-Based Hostel Search")
    st.markdown("*UC-STU-001: Find hostels near your current location using GPS coordinates.*")

    st.info(
        "💡 **Intelligence:** GPS proximity is a **soft score** (not a hard cutoff). "
        "A highly-recommended hostel 5km away can still outrank a poor hostel 0.5km away. "
        "Final score = 60% Hybrid + 40% Proximity"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📌 Your Location")
        location_preset = st.selectbox(
            "Quick preset",
            ["FAST NUCES H-11 Campus",
             "F-10 Markaz",
             "G-11 Markaz",
             "Custom Location"]
        )

        presets = {
            "FAST NUCES H-11 Campus": (33.6461, 72.9928),
            "F-10 Markaz"           : (33.7050, 72.9693),
            "G-11 Markaz"           : (33.6710, 72.9801),
            "Custom Location"       : (33.6461, 72.9928),
        }

        default_lat, default_lng = presets[location_preset]
        lat = st.number_input("Latitude",  value=default_lat, format="%.4f")
        lng = st.number_input("Longitude", value=default_lng, format="%.4f")

        gender_gps = st.selectbox("Your Gender", ["Male","Female"])
        top_k_gps  = st.slider("Number of Results", 3, 10, 5)

    with col2:
        st.markdown("### ⚙️ Quick Preferences")
        dept_gps    = st.selectbox("Department", [
            "Computer Science","Software Engineering",
            "Electrical Engineering","BBA","Social Sciences"
        ])
        budget_gps  = st.number_input("Max Budget (PKR)", 5000, 50000, 20000, 1000)
        study_gps   = st.slider("Study Focus", 0.0, 1.0, 0.6, 0.05)
        food_gps    = st.selectbox("Food Preference", ["Both","Veg","Non-Veg"])
        room_gps    = st.selectbox("Room Type", ["Single","Double","Dormitory"])

    if st.button("📍 Search Nearby Hostels", type="primary", use_container_width=True):
        with st.spinner("Computing GPS distances and hybrid scores..."):
            hostel_type = "Girls" if gender_gps=="Female" else "Boys"
            hostels_sub = hostels_df[hostels_df["hostel_type"]==hostel_type].copy()
            hostels_sub["gps_dist"] = hostels_sub.apply(
                lambda h: haversine_distance(lat,lng,h["latitude"],h["longitude"]),
                axis=1
            )
            hostels_sub["proximity"] = np.exp(-0.3*hostels_sub["gps_dist"])

            # Get hybrid scores via cold-start
            stype = classify_student_type(study_gps, 0.6, 0.5)
            alpha = type_alphas.get(stype, best_alpha)
            model = cold_start_models.get(gender_gps)
            if model:
                feat = np.array([[budget_gps, 3.0, study_gps,
                                   0.6, 0.5, 0.3, 1, 0, 0, 0, 0]])
                scaled  = model["scaler"].transform(feat)
                cluster = model["kmeans"].predict(scaled)[0]
                labels  = model["kmeans"].labels_
                sim_idx = [model["indices"][i] for i,lbl in enumerate(labels) if lbl==cluster]
                sim_sids= students_df.loc[sim_idx,"student_id"].tolist()
                avg     = predicted_matrix.loc[predicted_matrix.index.isin(sim_sids)].mean(axis=0)
                cf_s    = avg[avg.index.isin(hostels_sub["hostel_id"].values)]
                if cf_s.max()>cf_s.min():
                    cf_s = (cf_s-cf_s.min())/(cf_s.max()-cf_s.min())
                hostels_sub["hybrid"] = hostels_sub["hostel_id"].map(cf_s).fillna(0)
            else:
                hostels_sub["hybrid"] = hostels_sub["overall_rating"]/5

            hostels_sub["final_score"] = (
                0.60*hostels_sub["hybrid"] +
                0.40*hostels_sub["proximity"]
            )
            results = hostels_sub.nlargest(top_k_gps, "final_score")

        st.success(f"✅ Found {len(results)} nearby hostels")

        # Map visualisation
        st.markdown("### 🗺️ Hostel Map")
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.scatter(
            hostels_sub["longitude"], hostels_sub["latitude"],
            c="#bdc3c7", s=40, alpha=0.5, label="All hostels", zorder=2
        )
        top_h = hostels_df[hostels_df["hostel_id"].isin(results["hostel_id"])]
        sc = ax.scatter(
            top_h["longitude"], top_h["latitude"],
            c=results["final_score"].values,
            cmap="RdYlGn", s=150, zorder=4,
            label="Top recommendations"
        )
        ax.scatter(lng, lat, c="#f39c12", s=300,
                   marker="*", zorder=5, label="Your location")
        for _, row in top_h.iterrows():
            r_row = results[results["hostel_id"]==row["hostel_id"]]
            score = r_row["final_score"].values[0] if len(r_row)>0 else 0
            ax.annotate(
                f"{row['hostel_name'][:18]}\n({score:.2f})",
                (row["longitude"], row["latitude"]),
                textcoords="offset points", xytext=(6,6), fontsize=7
            )
        plt.colorbar(sc, ax=ax, label="Final Score", shrink=0.8)
        ax.set_title(f"GPS Search Results — {hostel_type} Hostels Near You",
                     fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Results table
        st.markdown("### 📋 Results")
        for rank, (_, row) in enumerate(results.iterrows(), 1):
            col_a, col_b, col_c, col_d, col_e = st.columns([3,1,1,1,1])
            col_a.markdown(f"**#{rank} {row['hostel_name']}** — {row['area']}")
            col_b.metric("Distance", f"{row['gps_dist']:.2f}km")
            col_c.metric("Proximity", f"{row['proximity']:.3f}")
            col_d.metric("Hybrid", f"{row['hybrid']:.3f}")
            col_e.metric("Final", f"{row['final_score']:.3f}")


# ══════════════════════════════════════════════════════════════════
# PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Report")
    st.markdown(
        "*Evaluated on a held-out test set (80/20 train/test split). "
        "Models were trained on 80% of data, evaluated on the remaining 20% they never saw.*"
    )

    # Metrics table
    st.markdown("### 📈 Final Metrics Comparison")
    metrics_data = {
        "Metric"           : ["Precision@3","Precision@5","MAP","Coverage","RMSE"],
        "Content-Based"    : [
            hybrid_metrics["Content-Based"]["P@3"],
            hybrid_metrics["Content-Based"]["P@5"],
            hybrid_metrics["Content-Based"]["MAP"],
            hybrid_metrics["Content-Based"]["Coverage"],
            "N/A"
        ],
        "Collaborative(SVD)":[
            hybrid_metrics["Collaborative"]["P@3"],
            hybrid_metrics["Collaborative"]["P@5"],
            hybrid_metrics["Collaborative"]["MAP"],
            hybrid_metrics["Collaborative"]["Coverage"],
            cf_metrics.get("RMSE","N/A")
        ],
        "🏆 Hybrid"        : [
            hybrid_metrics["Hybrid"]["P@3"],
            hybrid_metrics["Hybrid"]["P@5"],
            hybrid_metrics["Hybrid"]["MAP"],
            hybrid_metrics["Hybrid"]["Coverage"],
            0.3876
        ],
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # CB vs CF vs Hybrid bar chart
        fig, ax = plt.subplots(figsize=(8,5))
        metrics_plot = ["P@3","P@5","MAP","Coverage"]
        cb_v = [hybrid_metrics["Content-Based"][m] for m in metrics_plot]
        cf_v = [hybrid_metrics["Collaborative"][m] for m in metrics_plot]
        hy_v = [hybrid_metrics["Hybrid"][m] for m in metrics_plot]
        x,w  = np.arange(len(metrics_plot)), 0.25
        b1 = ax.bar(x-w, cb_v, w, label="Content-Based",    color="#3498db", edgecolor="white")
        b2 = ax.bar(x,   cf_v, w, label="Collaborative",    color="#2ecc71", edgecolor="white")
        b3 = ax.bar(x+w, hy_v, w, label="Hybrid ★",         color="#e74c3c", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_plot)
        ax.set_title("CB vs CF vs Hybrid", fontweight="bold")
        ax.set_ylabel("Score")
        ax.set_ylim(0,1.1)
        ax.legend()
        ax.grid(axis="y",alpha=0.3)
        for bar in list(b1)+list(b2)+list(b3):
            h=bar.get_height()
            if h>0.01:
                ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                        f"{h:.3f}", ha="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        # Alpha tuning curve
        fig, ax = plt.subplots(figsize=(8,5))
        alpha_res = hybrid_config["alpha_results"]
        alphas    = [float(k) for k in alpha_res.keys()]
        maps      = list(alpha_res.values())
        ax.plot(alphas, maps, "o-", color="#9b59b6", linewidth=2, markersize=8)
        ax.axvline(best_alpha, color="red", linestyle="--",
                   label=f"Optimal α={best_alpha}")
        ax.fill_between(alphas, maps, alpha=0.15, color="#9b59b6")
        ax.set_title("Alpha (α) Tuning — LEARNED Not Hardcoded", fontweight="bold")
        ax.set_xlabel("α (0=pure CF, 1=pure CB)")
        ax.set_ylabel("Validation MAP")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("### 🧠 Intelligence Proof")

    proof_cols = st.columns(3)
    with proof_cols[0]:
        st.success("**α = 0.3 LEARNED**\nNot hardcoded — found by testing 11 values on validation data")
    with proof_cols[1]:
        st.success("**SVD Latent Factors**\nDiscovers hidden preference patterns no human defined")
    with proof_cols[2]:
        st.success("**Hybrid > Both**\nP@3=0.1958 and MAP=0.2754 beat both CB and CF individually")

    proof_cols2 = st.columns(3)
    with proof_cols2[0]:
        st.info("**Time-Decay Weighting**\nRecent interactions weighted more than old ones")
    with proof_cols2[1]:
        st.info("**Adaptive Alpha**\n4 student types each get personalised CB/CF balance")
    with proof_cols2[2]:
        st.info("**Cold Start Solved**\nNew students handled via 8-cluster demographic grouping")

    st.divider()
    st.markdown("### ⚙️ Adaptive Alpha per Student Type")
    ta_data = {
        "Student Type"  : list(type_alphas.keys()),
        "α (CB weight)" : list(type_alphas.values()),
        "CB %"          : [f"{v:.0%}" for v in type_alphas.values()],
        "CF %"          : [f"{1-v:.0%}" for v in type_alphas.values()],
        "Students"      : [
            len(students_df[students_df.apply(
                lambda s: classify_student_type(
                    s["study_preference"],
                    s["price_sensitivity"],
                    s["comfort_preference"]
                )==t, axis=1
            )])
            for t in type_alphas.keys()
        ]
    }
    st.dataframe(pd.DataFrame(ta_data), use_container_width=True)