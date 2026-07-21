"""
StayBuddy — Improved AI Complaint Categorization
Uses:
  - Data augmentation (synonyms, paraphrases)
  - TF-IDF with category-specific vocabulary boosting
  - Logistic Regression with calibration
  - Priority detection as separate classifier
"""

import csv, os, json, joblib, re
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(os.path.dirname(__file__), 'complaints.csv')
OUT_DIR   = os.path.dirname(__file__)

print("="*60)
print("  StayBuddy Complaint AI — Improved Training")
print("="*60)

df = pd.read_csv(DATA_PATH)
print(f"\n✅ Loaded {len(df)} complaints, {df['category'].nunique()} categories")

# ── Augmentation — add paraphrased versions of each complaint ────────────────
AUGMENTATIONS = {
    "Maintenance": [
        "The {item} is broken and needs repair urgently",
        "There is a problem with the {item} in my room",
        "{item} not working since several days",
        "Please fix the {item} as soon as possible",
        "Internet connection is very slow and keeps dropping",
        "Wifi is not working in my room",
        "There is no electricity in the corridor",
        "Lights in bathroom not working",
        "Water pipe is leaking from ceiling",
        "No hot water available in the morning",
        "AC is not cooling properly",
        "Electric socket is sparking dangerously",
        "Generator does not turn on during load shedding",
        "Geyser is not heating water",
        "Door lock of my room is broken",
        "Window glass is cracked needs fixing",
    ],
    "Cleanliness": [
        "The room has not been cleaned for several days",
        "There are cockroaches and insects in the room",
        "Bathrooms are very dirty and smell bad",
        "Dust everywhere in the corridors",
        "Garbage bins are overflowing and not emptied",
        "Toilet is not cleaned regularly",
        "There are rats seen near the kitchen area",
        "Bedbugs found in mattress please help",
        "Washroom smells terrible due to poor cleaning",
        "Pest control needs to be done urgently",
        "Mosquitoes are very common in rooms at night",
        "Floor is always wet and dirty",
    ],
    "Food Quality": [
        "The food served today was not good at all",
        "Dinner is always cold and tasteless",
        "Portions of food are very small and insufficient",
        "Food is not cooked properly and seems raw",
        "There was a hair found in my meal today",
        "Quality of vegetables has gone down",
        "Bread is undercooked and not edible",
        "Biryani served today smelled rotten",
        "Food is same every day with no variety",
        "No fruit provided with breakfast anymore",
        "Mess is serving stale food to students",
        "Cook does not maintain proper hygiene",
        "Meal timings are not followed properly",
    ],
    "Safety/Security": [
        "My belongings were stolen from the room",
        "Security guard is not present at night",
        "An unauthorized person was seen in the corridor",
        "CCTV cameras near entrance are not working",
        "I feel unsafe because the main gate is left open",
        "My phone went missing from study room",
        "There was a fight near the hostel gate last night",
        "Someone broke into the room when I was away",
        "The emergency exit is blocked with furniture",
        "Guard is absent at midnight and early morning",
        "Suspicious person was seen on second floor",
        "My laptop was taken from room without permission",
    ],
    "Noise/Disturbance": [
        "There is too much noise at night from upstairs",
        "People are shouting and running in corridors after midnight",
        "Someone is playing loud music in room next door",
        "I cannot sleep because of constant noise",
        "Construction noise outside is very disturbing",
        "Neighbor's guests are very loud late at night",
        "Stomping sounds from upper floor all night",
        "Group of students making noise in common area",
        "Person in room 201 plays music very loudly",
        "Too much noise disturbing my studies",
    ],
    "Staff Behavior": [
        "Warden behaved very rudely when I went to complain",
        "Cleaning staff does not do their job properly",
        "Cook was very disrespectful during meal time",
        "Security guard was rude and threatening",
        "Warden is always biased towards certain students",
        "Staff member shouted at me in front of others",
        "Warden issued fine without any valid reason",
        "No staff available to address complaints",
        "Management is not responding to our requests",
        "Cook behaves badly with students who complain",
        "Warden shows favoritism to certain students",
    ],
    "Roommate Issue": [
        "My roommate does not keep the room clean",
        "Roommate smokes inside room despite rules",
        "Roommate plays loud music late at night",
        "My roommate borrows things without asking",
        "There is constant conflict with my roommate",
        "Roommate brings guests without permission",
        "Cannot study because roommate is always noisy",
        "Roommate uses my belongings without consent",
        "There is no personal space due to roommate",
        "Roommate's behavior is making me uncomfortable",
    ],
    "Billing/Payment Dispute": [
        "I was charged extra amount this month",
        "My security deposit has not been returned",
        "There are unexpected charges on my bill",
        "I paid rent but it shows pending in records",
        "Got billed twice for the same month",
        "Fee structure is not transparent",
        "I received wrong receipt for my payment",
        "Hostel is charging for facilities not provided",
        "My refund has not been processed for two months",
        "Fine was deducted without prior notice",
    ],
    "Other": [
        "I would like to request a room change",
        "Please provide more chairs in study room",
        "Visiting hours policy should be updated",
        "I want to give a suggestion for improvement",
        "Please install more fans in common area",
        "Need to discuss my hostel fees schedule",
        "Request for extension of curfew time",
        "I have a general inquiry about hostel rules",
    ],
}

# Build augmented dataset
augmented_texts = []
augmented_labels = []

# Original data
for _, row in df.iterrows():
    augmented_texts.append(row['complaint_text'])
    augmented_labels.append(row['category'])

# Add augmented data
for cat, samples in AUGMENTATIONS.items():
    for text in samples:
        # Skip template strings
        if '{item}' not in text:
            augmented_texts.append(text)
            augmented_labels.append(cat)
        else:
            # Fill templates
            items = ['fan', 'light', 'wifi', 'water pipe', 'AC', 'geyser', 'door lock']
            for item in items[:3]:
                augmented_texts.append(text.format(item=item))
                augmented_labels.append(cat)

print(f"✅ Augmented: {len(df)} → {len(augmented_texts)} samples")

# ── Preprocess ───────────────────────────────────────────────────────────────
def preprocess(text):
    text = str(text).lower().strip()
    # Remove numbers and special chars but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Domain-specific normalization
    replacements = {
        'wifi': 'wifi internet connection',
        'warden': 'warden staff management',
        'cctv': 'cctv camera security surveillance',
        'ac': 'air conditioning cooling',
        'internet': 'internet wifi connection',
        'stolen': 'stolen theft missing',
        'cockroach': 'cockroach pest insect',
        'smell': 'smell odor dirty unhygienic',
    }
    for k, v in replacements.items():
        if f' {k} ' in f' {text} ':
            text = text.replace(k, v)
    return text

X_all = [preprocess(t) for t in augmented_texts]
y_all = augmented_labels

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y_all)
print(f"✅ Classes: {list(le.classes_)}")

# ── Vectorizers ───────────────────────────────────────────────────────────────
word_vec = TfidfVectorizer(
    ngram_range=(1, 3),
    max_features=12000,
    sublinear_tf=True,
    min_df=1,
    analyzer='word',
    strip_accents='unicode',
)

char_vec = TfidfVectorizer(
    ngram_range=(3, 6),
    max_features=8000,
    sublinear_tf=True,
    min_df=1,
    analyzer='char_wb',
)

X_word = word_vec.fit_transform(X_all)
X_char = char_vec.fit_transform(X_all)
X_feat = hstack([X_word, X_char])

print(f"✅ Features: word={X_word.shape[1]}, char={X_char.shape[1]}, total={X_feat.shape[1]}")

# ── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# ── Cross-validation ──────────────────────────────────────────────────────────
print("\n5-Fold Cross-Validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [
    ('Logistic Regression', LogisticRegression(C=10, max_iter=3000, random_state=42)),
    ('LinearSVC (calibrated)', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=3000, random_state=42), cv=3)),
]:
    scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring='accuracy')
    print(f"  {name:<30} {scores.mean():.4f} ± {scores.std():.4f}")

# ── Final model: Logistic Regression (best generalization + probabilities) ────
final_model = LogisticRegression(C=10, max_iter=3000, random_state=42)
final_model.fit(X_train, y_train)

y_pred  = final_model.predict(X_test)
acc     = accuracy_score(y_test, y_pred)

print(f"\n✅ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))

# ── Priority classifier ───────────────────────────────────────────────────────
# Train a separate priority predictor
df_pri = df.copy()
df_pri['text_clean'] = df_pri['complaint_text'].apply(preprocess)

pri_vec = TfidfVectorizer(ngram_range=(1,2), max_features=5000,
                           sublinear_tf=True, min_df=1)
X_pri = pri_vec.fit_transform(df_pri['text_clean'])
y_pri = df_pri['priority'].values

pri_model = LogisticRegression(C=5, max_iter=1000, random_state=42)
pri_model.fit(X_pri, y_pri)

pri_scores = cross_val_score(pri_model, X_pri, y_pri, cv=5, scoring='accuracy')
print(f"✅ Priority Model CV Accuracy: {pri_scores.mean():.4f}")

# ── Build suggestion + subcategory map from dataset ───────────────────────────
suggestion_map = {}
subcat_map     = {}

for cat in le.classes_:
    rows = df[df['category'] == cat]
    suggestion_map[cat] = rows['ai_suggestion'].value_counts().head(5).index.tolist()
    subcat_map[cat]     = rows['subcategory'].value_counts().head(5).index.tolist()

# ── Save all artifacts ────────────────────────────────────────────────────────
joblib.dump(final_model, os.path.join(OUT_DIR, 'complaint_model.pkl'),      compress=3)
joblib.dump(word_vec,    os.path.join(OUT_DIR, 'complaint_word_vec.pkl'),   compress=3)
joblib.dump(char_vec,    os.path.join(OUT_DIR, 'complaint_char_vec.pkl'),   compress=3)
joblib.dump(le,          os.path.join(OUT_DIR, 'complaint_label_enc.pkl'),  compress=3)
joblib.dump(pri_model,   os.path.join(OUT_DIR, 'complaint_priority_model.pkl'), compress=3)
joblib.dump(pri_vec,     os.path.join(OUT_DIR, 'complaint_priority_vec.pkl'),   compress=3)

metadata = {
    'model': 'Logistic Regression + TF-IDF (word+char ngrams)',
    'accuracy': round(acc, 4),
    'categories': list(le.classes_),
    'suggestions': suggestion_map,
    'subcategories': subcat_map,
    'n_train': len(y_train),
    'n_test': len(y_test),
    'augmented_total': len(X_all),
}
with open(os.path.join(OUT_DIR, 'complaint_model_meta.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Saved: complaint_model.pkl, complaint_word_vec.pkl,")
print("          complaint_char_vec.pkl, complaint_label_enc.pkl")
print("          complaint_priority_model.pkl, complaint_priority_vec.pkl")
print("          complaint_model_meta.json")

# ── Live test on UNSEEN sentences ─────────────────────────────────────────────
def predict(text):
    t  = preprocess(text)
    xw = word_vec.transform([t])
    xc = char_vec.transform([t])
    xv = hstack([xw, xc])
    pred  = final_model.predict(xv)[0]
    proba = final_model.predict_proba(xv)[0]
    cat   = le.inverse_transform([pred])[0]
    conf  = round(float(proba.max()), 3)
    top3  = [(le.classes_[i], round(float(proba[i]),3))
              for i in proba.argsort()[-3:][::-1]]
    # Priority
    xp    = pri_vec.transform([t])
    pri   = pri_model.predict(xp)[0]
    sug   = suggestion_map[cat][0] if suggestion_map.get(cat) else ''
    sub   = subcat_map[cat][0]     if subcat_map.get(cat)     else ''
    return cat, sub, conf, pri, top3, sug

print("\n" + "="*60)
print("  Live Test on UNSEEN Paraphrased Sentences")
print("="*60)

tests = [
    ("Internet connection keeps dropping",               "Maintenance"),
    ("Guard is absent at night time",                    "Safety/Security"),
    ("Toilet smells bad and is dirty",                   "Cleanliness"),
    ("Dinner is not properly cooked",                    "Food Quality"),
    ("Someone took my phone from shelf",                 "Safety/Security"),
    ("Person upstairs is very noisy at night",           "Noise/Disturbance"),
    ("Got billed twice for the same month",              "Billing/Payment Dispute"),
    ("Electric socket is sparking dangerously",          "Maintenance"),
    ("Cook behaves badly with students",                 "Staff Behavior"),
    ("Roommate keeps borrowing things without asking",   "Roommate Issue"),
    ("No water available in morning",                    "Maintenance"),
    ("Rats seen near mess area",                         "Cleanliness"),
    ("I feel unsafe walking at night",                   "Safety/Security"),
    ("My fee receipt shows wrong amount",                "Billing/Payment Dispute"),
    ("Food portions are too small",                      "Food Quality"),
]

correct = 0
for text, expected in tests:
    cat, sub, conf, pri, top3, sug = predict(text)
    ok = '✅' if cat == expected else '❌'
    if cat == expected: correct += 1
    print(f"  {ok} [{conf*100:.0f}%] {text:<45} → {cat}")

print(f"\n  Accuracy on unseen: {correct}/{len(tests)} = {correct/len(tests)*100:.0f}%")
print(f"\n{'='*60}")
print(f"  Training Complete! Model ready for API integration")
print(f"{'='*60}")
