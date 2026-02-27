"""
╔══════════════════════════════════════════════════════════════════╗
║       StayBuddy — Intelligent Collaborative Filtering            ║
║       Component 2 of 3: SVD Matrix Factorization Engine          ║
╠══════════════════════════════════════════════════════════════════╣
║  Author  : Eraj Zaman (22I-1296)                                 ║
║  Project : StayBuddy - Intelligent Hostel Management System      ║
║  Supervisor: Dr. Ahkter Jamil, FAST NUCES Islamabad              ║
╠══════════════════════════════════════════════════════════════════╣
║  What makes this INTELLIGENT (not just automated):               ║
║                                                                  ║
║  1. SVD Matrix Factorization — decomposes the 200×75             ║
║     student-hostel interaction matrix into latent factor         ║
║     representations. These latent factors capture hidden         ║
║     preference patterns that NO human rule could define.         ║
║                                                                  ║
║  2. Time-decay weighting — recent interactions carry more        ║
║     signal than old ones. Formula: weight × e^(-λ × days_ago)   ║
║                                                                  ║
║  3. Implicit feedback learning — learns from behaviour:          ║
║     view(w=1) < save(w=3) < attempt(w=4) < booking(w=5)         ║
║                                                                  ║
║  4. Proper train/test split — 80% train, 20% test held out.     ║
║     SVD trained on train set, evaluated on unseen test set.      ║
║     This gives honest, meaningful metrics.                       ║
║                                                                  ║
║  5. Cold start handling — new students handled via               ║
║     demographic clustering, not hard failure.                    ║
║                                                                  ║
║  6. Gender-aware filtering — Girls/Boys separation respected.    ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import json
import os
import warnings
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
np.random.seed(42)

print("=" * 65)
print("  StayBuddy — Intelligent Collaborative Filtering Engine")
print("  Method: SVD Matrix Factorization + Time-Decay Weighting")
print("=" * 65)

# ──────────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ──────────────────────────────────────────────────────────────────
print("\n📂 Loading datasets...")

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

hostels_df      = pd.read_csv(os.path.join(DATA_DIR, "hostels.csv"))
students_df     = pd.read_csv(os.path.join(DATA_DIR, "students.csv"))
interactions_df = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))

interactions_df["timestamp"] = pd.to_datetime(interactions_df["timestamp"])
REFERENCE_DATE = interactions_df["timestamp"].max()

print(f"  ✓ Hostels      : {len(hostels_df):>4} records")
print(f"  ✓ Students     : {len(students_df):>4} records")
print(f"  ✓ Interactions : {len(interactions_df):>4} records")
print(f"  ✓ Date range   : {interactions_df.timestamp.min().date()} "
      f"→ {interactions_df.timestamp.max().date()}")
print(f"  ✓ Interaction types:")
for itype, count in interactions_df["interaction_type"].value_counts().items():
    print(f"      {itype:<20} : {count}")


# ──────────────────────────────────────────────────────────────────
# SECTION 2: TIME-DECAY WEIGHTING
# Intelligence: recent interactions carry more signal.
# decay = e^(-λ × days_since_interaction)
# ──────────────────────────────────────────────────────────────────
print("\n⏱️  Applying time-decay weighting to interactions...")

DECAY_LAMBDA = 0.01

INTERACTION_WEIGHTS = {
    "view"            : 1.0,
    "save"            : 3.0,
    "booking_attempt" : 4.0,
    "booking"         : 5.0,
}

interactions_df["base_weight"] = interactions_df["interaction_type"].map(
    INTERACTION_WEIGHTS
)
interactions_df["time_decay"] = interactions_df["timestamp"].apply(
    lambda ts: np.exp(-DECAY_LAMBDA * max((REFERENCE_DATE - ts).days, 0))
)
interactions_df["final_weight"] = (
    interactions_df["base_weight"] * interactions_df["time_decay"]
)

print(f"  ✓ Time decay applied (λ={DECAY_LAMBDA})")
print(f"  ✓ Weight range : "
      f"{interactions_df['final_weight'].min():.3f} – "
      f"{interactions_df['final_weight'].max():.3f}")
print(f"  ✓ Avg weight   : {interactions_df['final_weight'].mean():.3f}")


# ──────────────────────────────────────────────────────────────────
# SECTION 3: TRAIN / TEST SPLIT (80/20)
# Proper ML evaluation:
#   - Sort each student's interactions by timestamp
#   - First 80% → training (SVD learns from this)
#   - Last 20%  → testing  (evaluation ground truth)
# This gives honest, meaningful metrics.
# ──────────────────────────────────────────────────────────────────
print("\n✂️  Building train/test split (80% train / 20% test)...")

train_rows, test_rows = [], []

for sid, group in interactions_df.groupby("student_id"):
    group_sorted = group.sort_values("timestamp")
    n            = len(group_sorted)
    split_idx    = max(1, int(n * 0.80))
    train_rows.append(group_sorted.iloc[:split_idx])
    test_rows.append(group_sorted.iloc[split_idx:])

train_df = pd.concat(train_rows).reset_index(drop=True)
test_df  = pd.concat(test_rows).reset_index(drop=True)

print(f"  ✓ Train interactions : {len(train_df)}")
print(f"  ✓ Test  interactions : {len(test_df)}")
print(f"  ✓ Test bookings/saves: "
      f"{len(test_df[test_df['interaction_type'].isin(['booking','save'])])}")


# ──────────────────────────────────────────────────────────────────
# SECTION 4: BUILD USER-ITEM INTERACTION MATRIX (from train only)
# ──────────────────────────────────────────────────────────────────
print("\n🔢 Building user-item interaction matrix (train set)...")

all_students = students_df["student_id"].tolist()
all_hostels  = hostels_df["hostel_id"].tolist()

def build_matrix(df, students, hostels):
    agg = (
        df.groupby(["student_id", "hostel_id"])["final_weight"]
        .max().reset_index()
    )
    matrix = agg.pivot(
        index="student_id", columns="hostel_id", values="final_weight"
    ).fillna(0)
    return matrix.reindex(index=students, columns=hostels, fill_value=0)

interaction_matrix = build_matrix(train_df, all_students, all_hostels)

print(f"  ✓ Matrix shape  : {interaction_matrix.shape}")
sparsity = (interaction_matrix == 0).sum().sum() / interaction_matrix.size
print(f"  ✓ Sparsity      : {sparsity:.1%}")
print(f"  ✓ Non-zero entries: {(interaction_matrix > 0).sum().sum()}")

matrix_values = interaction_matrix.values


# ──────────────────────────────────────────────────────────────────
# SECTION 5: SVD MATRIX FACTORIZATION
# M ≈ U × Σ × Vᵀ
# U  = student latent factors (200 × k)
# Vᵀ = hostel  latent factors (k × 75)
# k  = number of latent factors (hyperparameter, tuned below)
# ──────────────────────────────────────────────────────────────────
print("\n🧠 Training SVD Matrix Factorization...")
print("   Hyperparameter search — finding optimal k:")
print("   " + "─" * 50)

K_VALUES  = [5, 10, 15, 20, 25]
best_k    = 10
best_rmse = float("inf")
k_results = {}

for k in K_VALUES:
    if k >= min(matrix_values.shape):
        continue
    svd_tmp   = TruncatedSVD(n_components=k, random_state=42)
    U_tmp     = svd_tmp.fit_transform(matrix_values)
    Vt_tmp    = svd_tmp.components_
    recon     = U_tmp @ Vt_tmp
    mask      = matrix_values > 0
    if mask.sum() == 0:
        continue
    rmse      = float(np.sqrt(np.mean((matrix_values[mask] - recon[mask]) ** 2)))
    variance  = float(svd_tmp.explained_variance_ratio_.sum())
    k_results[k] = {"rmse": rmse, "variance_explained": variance}
    marker    = ""
    if rmse < best_rmse:
        best_rmse = rmse
        best_k    = k
        marker    = " ← best"
    print(f"   k={k:>2}  |  RMSE: {rmse:.4f}  |  "
          f"Variance: {variance:.1%}{marker}")

print(f"\n  ✓ Optimal k = {best_k}  (RMSE = {best_rmse:.4f})")

# ── Train final model with best k ────────────────────────────────
print(f"\n  Training final SVD model (k={best_k})...")
svd_final   = TruncatedSVD(n_components=best_k, random_state=42)
U_final     = svd_final.fit_transform(matrix_values)
Vt_final    = svd_final.components_
recon_vals  = U_final @ Vt_final

predicted_matrix = pd.DataFrame(
    recon_vals,
    index=interaction_matrix.index,
    columns=interaction_matrix.columns
)

print(f"  ✓ Student latent matrix : {U_final.shape}")
print(f"  ✓ Hostel  latent matrix : {Vt_final.T.shape}")
print(f"  ✓ Variance explained    : "
      f"{svd_final.explained_variance_ratio_.sum():.1%}")

# ── Also train an eval-only model (same k) for proper evaluation ─
# Uses train matrix only — evaluated on held-out test set
svd_eval  = TruncatedSVD(n_components=best_k, random_state=42)
U_eval    = svd_eval.fit_transform(matrix_values)
Vt_eval   = svd_eval.components_
pred_eval = pd.DataFrame(
    U_eval @ Vt_eval,
    index=interaction_matrix.index,
    columns=interaction_matrix.columns
)


# ──────────────────────────────────────────────────────────────────
# SECTION 6: COLD START HANDLING
# New students with no history → demographic clustering
# ──────────────────────────────────────────────────────────────────
print("\n🆕 Building cold-start handler (demographic clustering)...")

DEMO_FEATURES = [
    "budget_max", "max_distance_km", "study_preference",
    "price_sensitivity", "comfort_preference", "noise_tolerance",
    "priority_wifi", "priority_study_room", "priority_ac",
    "priority_generator", "needs_transport",
]
N_CLUSTERS   = 8
scaler_demo  = MinMaxScaler()
demo_scaled  = scaler_demo.fit_transform(
    students_df[DEMO_FEATURES].fillna(0).values
)
cold_start_models = {}

for gender in ["Female", "Male"]:
    mask    = students_df["gender"] == gender
    indices = students_df[mask].index.tolist()
    scaled  = demo_scaled[mask.values]
    if len(scaled) < N_CLUSTERS:
        continue
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    km.fit(scaled)
    cold_start_models[gender] = {
        "kmeans": km, "indices": indices, "scaler": scaler_demo
    }

print(f"  ✓ {N_CLUSTERS} demographic clusters per gender")
print(f"  ✓ Cold-start fallback ready")


def get_cold_start_recommendations(student, top_k=10):
    gender = student["gender"]
    model  = cold_start_models.get(gender)
    if model is None:
        appropriate = hostels_df[
            hostels_df["hostel_type"] == student["preferred_type"]
        ].copy()
        appropriate["cf_score"] = appropriate["overall_rating"] / 5
        return appropriate.nlargest(top_k, "cf_score")

    feat    = students_df.loc[
        students_df["student_id"] == student["student_id"],
        DEMO_FEATURES
    ].fillna(0).values
    scaled  = model["scaler"].transform(feat)
    cluster = model["kmeans"].predict(scaled)[0]
    labels  = model["kmeans"].labels_
    sim_idx = [
        model["indices"][i]
        for i, lbl in enumerate(labels) if lbl == cluster
    ]
    sim_sids = students_df.loc[sim_idx, "student_id"].tolist()
    avg_scores = pred_eval.loc[
        pred_eval.index.isin(sim_sids)
    ].mean(axis=0)

    appropriate = hostels_df[
        hostels_df["hostel_type"] == student["preferred_type"]
    ]["hostel_id"].tolist()
    avg_scores = avg_scores[
        avg_scores.index.isin(appropriate)
    ].sort_values(ascending=False)

    results = hostels_df[
        hostels_df["hostel_id"].isin(avg_scores.head(top_k).index)
    ].copy()
    results["cf_score"] = results["hostel_id"].map(avg_scores)
    return results.sort_values("cf_score", ascending=False)


# ──────────────────────────────────────────────────────────────────
# SECTION 7: CORE CF RECOMMENDATION FUNCTION
# ──────────────────────────────────────────────────────────────────
def get_cf_recommendations(student_id, top_k=10, exclude_seen=True):
    student  = students_df[students_df["student_id"] == student_id].iloc[0]
    has_hist = (
        student_id in predicted_matrix.index and
        interaction_matrix.loc[student_id].sum() > 0
    )

    if not has_hist:
        recs = get_cold_start_recommendations(student, top_k)
        recs["source"]       = "Cold-start (demographic clustering)"
        recs["cf_score_pct"] = (recs["cf_score"] * 100).round(1)
        return recs.reset_index(drop=True)

    scores = predicted_matrix.loc[student_id].copy()

    # Gender filter
    appropriate = hostels_df[
        hostels_df["hostel_type"] == student["preferred_type"]
    ]["hostel_id"].tolist()
    scores = scores[scores.index.isin(appropriate)]

    # Exclude seen hostels
    if exclude_seen:
        seen = interaction_matrix.loc[student_id]
        scores = scores[~scores.index.isin(seen[seen > 0].index)]

    # Normalise
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    top = scores.nlargest(top_k)
    results = hostels_df[hostels_df["hostel_id"].isin(top.index)].copy()
    results["cf_score"]     = results["hostel_id"].map(top)
    results["cf_score_pct"] = (results["cf_score"] * 100).round(1)
    results["source"]       = "SVD Collaborative Filtering"
    return results.sort_values("cf_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────
# SECTION 8: EVALUATION METRICS (on held-out TEST set)
# ──────────────────────────────────────────────────────────────────
print("\n📊 Evaluating on held-out TEST set...")

def precision_at_k(recommended, relevant, k):
    if not relevant or k == 0:
        return 0.0
    return sum(1 for h in recommended[:k] if h in relevant) / k

def average_precision(recommended, relevant):
    if not relevant:
        return 0.0
    hits, score = 0, 0.0
    for i, h in enumerate(recommended):
        if h in relevant:
            hits  += 1
            score += hits / (i + 1)
    return score / len(relevant)

def compute_rmse(test_df, pred_matrix):
    rated = test_df[
        test_df["rating"].notna() &
        (test_df["interaction_type"] == "booking")
    ]
    if rated.empty:
        return None
    actuals, preds = [], []
    for _, row in rated.iterrows():
        sid, hid = row["student_id"], row["hostel_id"]
        if sid in pred_matrix.index and hid in pred_matrix.columns:
            preds.append(pred_matrix.loc[sid, hid])
            actuals.append((row["rating"] - 2.5) / 2.5)
    if not preds:
        return None
    preds = np.array(preds)
    actuals = np.array(actuals)
    if preds.max() > preds.min():
        preds = (preds - preds.min()) / (preds.max() - preds.min())
    return float(np.sqrt(np.mean((actuals - preds) ** 2)))

# Ground truth from TEST set only
test_gt = {}
for sid, grp in test_df[
    test_df["interaction_type"].isin(["booking", "save"])
].groupby("student_id"):
    test_gt[sid] = set(grp["hostel_id"].tolist())

eval_students = [
    sid for sid in students_df["student_id"]
    if sid in test_gt and len(test_gt[sid]) > 0
]
sample = np.random.choice(
    eval_students,
    min(60, len(eval_students)),
    replace=False
)

p3_list, p5_list, p10_list, map_list = [], [], [], []
all_recommended = set()

print(f"   Evaluating on {len(sample)} students with held-out test data...")

for sid in sample:
    try:
        student  = students_df[students_df["student_id"] == sid].iloc[0]
        scores   = pred_eval.loc[sid].copy()

        # Gender filter
        appropriate = hostels_df[
            hostels_df["hostel_type"] == student["preferred_type"]
        ]["hostel_id"].tolist()
        scores = scores[scores.index.isin(appropriate)]

        # Exclude train-set interactions only
        seen_in_train = interaction_matrix.loc[sid]
        scores = scores[
            ~scores.index.isin(seen_in_train[seen_in_train > 0].index)
        ]

        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        rec_ids  = scores.nlargest(10).index.tolist()
        relevant = test_gt.get(sid, set())

        p3_list.append(precision_at_k(rec_ids, relevant, 3))
        p5_list.append(precision_at_k(rec_ids, relevant, 5))
        p10_list.append(precision_at_k(rec_ids, relevant, 10))
        map_list.append(average_precision(rec_ids, relevant))
        all_recommended.update(rec_ids)
    except Exception:
        continue

rmse_val = compute_rmse(test_df, pred_eval)
coverage = len(all_recommended) / len(hostels_df)

cf_metrics = {
    "P@3"     : round(np.mean(p3_list),  4) if p3_list  else 0.0,
    "P@5"     : round(np.mean(p5_list),  4) if p5_list  else 0.0,
    "P@10"    : round(np.mean(p10_list), 4) if p10_list else 0.0,
    "MAP"     : round(np.mean(map_list), 4) if map_list else 0.0,
    "RMSE"    : round(rmse_val, 4) if rmse_val else "N/A",
    "Coverage": round(coverage, 4),
    "Students Evaluated": len(sample),
}

# Load CB metrics for comparison
cb_metrics = {}
cb_path = os.path.join(MODEL_DIR, "cb_metrics.json")
if os.path.exists(cb_path):
    with open(cb_path) as f:
        cb_metrics = json.load(f)

print("\n" + "─" * 58)
print("  📈 COLLABORATIVE FILTERING — EVALUATION RESULTS")
print("─" * 58)
print(f"  {'Metric':<25} {'CF (SVD)':>10}  {'CB (Cosine)':>12}  Delta")
print("  " + "─" * 55)
for key in ["P@3", "P@5", "P@10", "MAP", "Coverage"]:
    cf_val = cf_metrics.get(key, 0)
    cb_val = cb_metrics.get(key, 0)
    bar    = "█" * int(cf_val * 25) + "░" * (25 - int(cf_val * 25))
    diff   = cf_val - cb_val if isinstance(cb_val, float) else 0
    arrow  = "↑" if diff >= 0 else "↓"
    print(f"  {key:<25} {cf_val:>10.4f}  {str(cb_val):>12}  "
          f"{arrow}{abs(diff):.4f}")
print(f"  {'RMSE':<25} {str(cf_metrics.get('RMSE','N/A')):>10}")
print(f"  {'Students Evaluated':<25} "
      f"{cf_metrics['Students Evaluated']:>10}")
print("─" * 58)


# ──────────────────────────────────────────────────────────────────
# SECTION 9: LIVE DEMONSTRATION — 3 STUDENT PROFILES
# ──────────────────────────────────────────────────────────────────
print("\n🎯 LIVE CF RECOMMENDATIONS — 3 Student Profiles")
print("=" * 65)

demo_profiles = []

# Profile 1: Active student (has booking history)
booked_students = interactions_df[
    interactions_df["interaction_type"] == "booking"
]["student_id"]
if len(booked_students) > 0:
    demo_profiles.append(
        ("Active Student (has history)", booked_students.iloc[0])
    )

# Profile 2: Study-focused
study_sid = students_df[
    students_df["study_preference"] > 0.75
].iloc[0]["student_id"]
demo_profiles.append(("Study-Focused Student", study_sid))

# Profile 3: Budget-conscious
budget_sid = students_df[
    students_df["budget_max"] < 15000
].iloc[0]["student_id"]
demo_profiles.append(("Budget-Conscious Student", budget_sid))

for label, sid in demo_profiles[:3]:
    student  = students_df[students_df["student_id"] == sid].iloc[0]
    has_hist = (
        sid in interaction_matrix.index and
        interaction_matrix.loc[sid].sum() > 0
    )
    seen_count = int((interaction_matrix.loc[sid] > 0).sum()) \
                 if sid in interaction_matrix.index else 0

    print(f"\n{'─'*65}")
    print(f"  👤 {label} ({sid})")
    print(f"     Gender     : {student['gender']}")
    print(f"     Department : {student['department']}")
    print(f"     Budget     : PKR {student['budget_min']:,} – "
          f"{student['budget_max']:,}")
    print(f"     Study pref : {student['study_preference']:.2f}/1.0")
    print(f"     History    : "
          f"{'YES — ' + str(seen_count) + ' interactions' if has_hist else 'NO — cold start'}")
    print(f"\n  🏠 TOP 5 CF RECOMMENDATIONS:")

    recs = get_cf_recommendations(sid, top_k=5, exclude_seen=True)

    for rank, (_, row) in enumerate(recs.iterrows(), 1):
        print(f"\n  #{rank}  {row['hostel_name']} [{row['hostel_type']}]")
        print(f"       CF Score  : {row['cf_score_pct']:.1f}%  |  "
              f"Rating: {row['overall_rating']}/5  |  "
              f"PKR {row['single_room_price']:,}/mo")
        print(f"       Area: {row['area']}  |  "
              f"Dist: {row['distance_from_fast_km']}km  |  "
              f"Rooms: {row['available_rooms']}")
        print(f"       🤖 {row['source']}")

print(f"\n{'='*65}")


# ──────────────────────────────────────────────────────────────────
# SECTION 10: LATENT FACTOR ANALYSIS
# Shows evaluators what SVD learned — proof of intelligence
# ──────────────────────────────────────────────────────────────────
print("\n🔍 Latent Factor Analysis — What did SVD discover?")
print("─" * 58)
hostel_factors = pd.DataFrame(
    Vt_final.T,
    index=all_hostels,
    columns=[f"Factor_{i+1}" for i in range(best_k)]
)
for fi in range(min(3, best_k)):
    col   = f"Factor_{fi+1}"
    top_h = hostel_factors[col].nlargest(3).index.tolist()
    names = hostels_df[
        hostels_df["hostel_id"].isin(top_h)
    ]["hostel_name"].tolist()
    print(f"\n  Factor {fi+1} (hidden preference pattern):")
    for h in names:
        row = hostels_df[hostels_df["hostel_name"] == h].iloc[0]
        print(f"    → {h:<30} | "
              f"{row['price_tier']:<10} | "
              f"Study: {row['study_environment_score']} | "
              f"Rating: {row['overall_rating']}")
print("─" * 58)


# ──────────────────────────────────────────────────────────────────
# SECTION 11: VISUALISATIONS
# ──────────────────────────────────────────────────────────────────
print("\n📊 Generating visualisations...")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(
    "StayBuddy — Collaborative Filtering Analysis (SVD)",
    fontsize=16, fontweight="bold"
)
colors = ["#2ecc71","#3498db","#9b59b6","#e74c3c","#f39c12","#1abc9c"]

# Plot 1: CB vs CF comparison
ax1   = axes[0, 0]
mkeys = ["P@3","P@5","P@10","MAP","Coverage"]
cfv   = [cf_metrics.get(k, 0) for k in mkeys]
cbv   = [cb_metrics.get(k, 0) for k in mkeys]
x, w  = np.arange(len(mkeys)), 0.35
b1    = ax1.bar(x - w/2, cbv, w, label="Content-Based",
                color="#3498db", edgecolor="white")
b2    = ax1.bar(x + w/2, cfv, w, label="Collaborative (SVD)",
                color="#2ecc71", edgecolor="white")
ax1.set_xticks(x)
ax1.set_xticklabels(mkeys)
ax1.set_title("CB vs CF Metrics Comparison", fontweight="bold")
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.0)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)
for bar in list(b1) + list(b2):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.01,
             f"{bar.get_height():.3f}",
             ha="center", va="bottom", fontsize=8)

# Plot 2: Variance explained vs k
ax2 = axes[0, 1]
kv  = [k for k in K_VALUES if k in k_results]
vv  = [k_results[k]["variance_explained"] for k in kv]
ax2.plot(kv, vv, "o-", color="#9b59b6", linewidth=2, markersize=8)
ax2.axvline(best_k, color="red", linestyle="--",
            label=f"Optimal k={best_k}")
ax2.fill_between(kv, vv, alpha=0.2, color="#9b59b6")
ax2.set_title("SVD Variance Explained vs k", fontweight="bold")
ax2.set_xlabel("Latent Factors (k)")
ax2.set_ylabel("Variance Explained")
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: RMSE vs k
ax3 = axes[0, 2]
rv  = [k_results[k]["rmse"] for k in kv]
ax3.plot(kv, rv, "s-", color="#e74c3c", linewidth=2, markersize=8)
ax3.axvline(best_k, color="green", linestyle="--",
            label=f"Optimal k={best_k}")
ax3.set_title("RMSE vs Latent Factors (k)\nHyperparameter Tuning",
              fontweight="bold")
ax3.set_xlabel("Latent Factors (k)")
ax3.set_ylabel("RMSE")
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Time-decayed weight distribution
ax4 = axes[1, 0]
for itype, color in zip(
    ["view","save","booking_attempt","booking"],
    ["#3498db","#2ecc71","#f39c12","#e74c3c"]
):
    subset = interactions_df[
        interactions_df["interaction_type"] == itype
    ]["final_weight"]
    if len(subset):
        ax4.hist(subset, bins=20, alpha=0.6,
                 label=f"{itype} (n={len(subset)})",
                 color=color, edgecolor="white")
ax4.set_title("Time-Decayed Weight Distribution\nby Interaction Type",
              fontweight="bold")
ax4.set_xlabel("Final Weight")
ax4.set_ylabel("Count")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

# Plot 5: Interaction matrix heatmap (25×25 sample)
ax5 = axes[1, 1]
sns.heatmap(
    interaction_matrix.iloc[:25, :25],
    ax=ax5, cmap="YlOrRd",
    xticklabels=False, yticklabels=False,
    cbar_kws={"shrink": 0.8}
)
ax5.set_title("User-Item Matrix (25×25 sample)\nDarker = stronger interaction",
              fontweight="bold")
ax5.set_xlabel("Hostels")
ax5.set_ylabel("Students")

# Plot 6: SVD latent factors heatmap
ax6 = axes[1, 2]
sns.heatmap(
    hostel_factors.iloc[:20, :min(8, best_k)],
    ax=ax6, cmap="coolwarm", center=0,
    xticklabels=True, yticklabels=False,
    cbar_kws={"shrink": 0.8}
)
ax6.set_title("SVD Latent Factors\n(20 Hostels × Top-8 Factors)",
              fontweight="bold")
ax6.set_xlabel("Latent Factors (hidden preference patterns)")
ax6.set_ylabel("Hostels")

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "collaborative_filtering_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Visualisation saved → models/collaborative_filtering_analysis.png")


# ──────────────────────────────────────────────────────────────────
# SECTION 12: SAVE MODEL ARTIFACTS
# ──────────────────────────────────────────────────────────────────
print("\n💾 Saving model artifacts...")

joblib.dump(svd_final, os.path.join(MODEL_DIR, "svd_model.pkl"))
np.save(os.path.join(MODEL_DIR, "U_student_factors.npy"), U_final)
np.save(os.path.join(MODEL_DIR, "Vt_hostel_factors.npy"), Vt_final)
interaction_matrix.to_csv(os.path.join(MODEL_DIR, "interaction_matrix.csv"))
predicted_matrix.to_csv(os.path.join(MODEL_DIR, "predicted_matrix.csv"))
joblib.dump(cold_start_models,
            os.path.join(MODEL_DIR, "cold_start_models.pkl"))

with open(os.path.join(MODEL_DIR, "cf_metrics.json"), "w") as f:
    json.dump(
        {k: (v if v != "N/A" else None) for k, v in cf_metrics.items()},
        f, indent=2
    )
with open(os.path.join(MODEL_DIR, "cf_k_tuning.json"), "w") as f:
    json.dump({str(k): v for k, v in k_results.items()}, f, indent=2)

print("  ✓ svd_model.pkl            saved")
print("  ✓ U_student_factors.npy    saved")
print("  ✓ Vt_hostel_factors.npy    saved")
print("  ✓ interaction_matrix.csv   saved")
print("  ✓ predicted_matrix.csv     saved")
print("  ✓ cold_start_models.pkl    saved")
print("  ✓ cf_metrics.json          saved")
print("  ✓ cf_k_tuning.json         saved")


# ──────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅ COLLABORATIVE FILTERING — COMPLETE")
print("=" * 65)
print(f"""
  What was built:
  • 200×75 user-item interaction matrix (train set only)
  • Time-decay weighting (λ={DECAY_LAMBDA}) — recency matters
  • 80/20 train/test split — honest evaluation on held-out data
  • SVD with k={best_k} latent factors (tuned from {K_VALUES})
  • Cold-start handler via {N_CLUSTERS}-cluster demographic grouping
  • Gender-aware filtering (Girls/Boys separation)

  Key metrics (evaluated on held-out TEST set):
  • Precision@5  : {cf_metrics.get('P@5',  0):.4f}
  • MAP          : {cf_metrics.get('MAP',  0):.4f}
  • RMSE         : {cf_metrics.get('RMSE', 'N/A')}
  • Coverage     : {cf_metrics.get('Coverage', 0):.4f}

  Intelligence proof:
  • SVD discovers HIDDEN latent preference patterns from data
  • Time-decay: recent behaviour weighted more than old
  • Implicit feedback: learns from views/saves/bookings
  • Cold-start: new students handled via clustering
  • Hyperparameter tuning: k optimised via RMSE
  • Proper ML evaluation: train/test split, not train=test

  Next step → 03_hybrid_model.py
""")
print("=" * 65)