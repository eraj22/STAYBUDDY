# similarity_scorer.py - FIXED VERSION

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from config import CLEANLINESS_MAP

STUDY_TEXT = {'studious': 0.85, 'moderate': 0.50, 'social': 0.15}

def _study_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.5
    s = str(val).strip().lower()
    if s in STUDY_TEXT:
        return STUDY_TEXT[s]
    try:
        return float(s)
    except:
        return 0.5


class SimilarityScorer:
    def __init__(self, feature_engineer):
        self.fe = feature_engineer
        self.model = None
        self.feature_names = None

    # ------------------------------------------------------------------
    def prepare_training_data(self, matches_df, profiles_dict):
        X, y = [], []
        for _, row in matches_df.iterrows():
            a, b, score = str(row['student_a']), str(row['student_b']), row['overall_score']
            if a in profiles_dict and b in profiles_dict:
                feats = self._extract_pair_features(profiles_dict[a], profiles_dict[b])
                if feats:
                    X.append(feats)
                    y.append(score)
        if not X:
            return np.array([]), np.array([])

        self.feature_names = [
            'study_sim','clean_sim','noise_sim','social_sim','curfew_sim',
            'price_sim','sleep_match','food_match','smoking_match','guest_match',
            'gender_match','dept_sim','year_sim','uni_match','budget_overlap','cultural_sim'
        ]
        return np.array(X), np.array(y)

    def train(self, X, y):
        if len(X) == 0:
            print("⚠️ No training data – rule-based matching will be used.")
            return None
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=50, max_depth=8,
                                           min_samples_split=5, random_state=42, n_jobs=-1)
        self.model.fit(X_tr, y_tr)
        y_pred = self.model.predict(X_te)
        print(f"📊 Model – MSE: {mean_squared_error(y_te, y_pred):.4f}  R²: {r2_score(y_te, y_pred):.4f}")
        return self.model

    # ------------------------------------------------------------------
    def _extract_pair_features(self, pa, pb):
        try:
            feats = []

            def num(p, key, default=0.5):
                return _study_float(p.get(key)) if key == 'study_habits' else _safe_float(p.get(key), default)

            def _safe_float(v, d=0.5):
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    return d
                try:
                    return float(v)
                except:
                    return d

            # Continuous similarity features
            for k in ['study_habits','noise_tolerance','social_preference','curfew_flexibility','price_sensitivity']:
                va = _study_float(pa.get(k)) if k == 'study_habits' else _safe_float(pa.get(k))
                vb = _study_float(pb.get(k)) if k == 'study_habits' else _safe_float(pb.get(k))
                feats.append(1 - min(1, abs(va - vb)))

            # Cleanliness
            def clean_num(p):
                v = p.get('cleanliness_level', 3)
                try:
                    return CLEANLINESS_MAP.get(int(float(v)), 0.5)
                except:
                    return 0.5
            feats.append(1 - abs(clean_num(pa) - clean_num(pb)))

            # Categorical matches
            feats.append(1 if pa.get('sleep_schedule') == pb.get('sleep_schedule') else 0)
            feats.append(1 if pa.get('food_preference') == pb.get('food_preference') else 0.5)
            feats.append(1 if pa.get('smoking_pref')    == pb.get('smoking_pref')    else 0)
            feats.append(1 if pa.get('guest_policy')    == pb.get('guest_policy')    else 0)
            feats.append(1 if pa.get('gender')          == pb.get('gender')          else 0)
            feats.append(1 if pa.get('department')      == pb.get('department') and pa.get('department') else 0)

            # Year similarity
            ya = _safe_float(pa.get('year'), 2)
            yb = _safe_float(pb.get('year'), 2)
            feats.append(1 - min(1, abs(ya - yb) / 4))

            feats.append(1 if pa.get('university') == pb.get('university') and pa.get('university') else 0)

            # Budget overlap
            mna = _safe_float(pa.get('budget_min'), 10000)
            mxa = _safe_float(pa.get('budget_max'), 20000)
            mnb = _safe_float(pb.get('budget_min'), 10000)
            mxb = _safe_float(pb.get('budget_max'), 20000)
            ol_min = max(mna, mnb); ol_max = min(mxa, mxb)
            if ol_max > ol_min:
                denom = min(mxa-mna, mxb-mnb) or 1
                feats.append(min(1, (ol_max - ol_min) / denom))
            else:
                feats.append(0)

            # Cultural
            ea = pa.get('ethnicity',''); eb = pb.get('ethnicity','')
            feats.append(1 if ea == eb and ea and ea != 'N/A' else 0)

            return feats
        except Exception as e:
            return None

    # ------------------------------------------------------------------
    def predict_similarity(self, profile_a, profile_b):
        if self.model is None:
            return self.fe.calculate_pairwise_similarity(profile_a, profile_b)
        feats = self._extract_pair_features(profile_a, profile_b)
        if feats is None:
            return self.fe.calculate_pairwise_similarity(profile_a, profile_b)
        pred = float(self.model.predict(np.array(feats).reshape(1, -1))[0])
        return round(min(1.0, max(0.0, pred)), 3)

    def save_model(self, path="room_matcher_model.pkl"):
        if self.model:
            joblib.dump({'model': self.model, 'features': self.feature_names}, path)
            print(f"✅ Model saved to {path}")

    def load_model(self, path="room_matcher_model.pkl"):
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                self.model = data['model']
                self.feature_names = data['features']
                print(f"✅ Model loaded from {path}")
                return True
            except Exception as e:
                print(f"⚠️  Could not load model ({e}), will retrain")
        return False