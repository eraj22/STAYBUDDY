# feature_engineering.py - FIXED VERSION

import pandas as pd
import numpy as np
from config import *

class FeatureEngineer:
    def __init__(self):
        pass

    def _safe_float(self, val, default=0.5):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    def _get_region(self, city):
        for region, cities in REGIONAL_CLUSTERS.items():
            if city in cities:
                return region
        return 'Other'

    def extract_features(self, profile):
        f = {}
        f['university']           = profile.get('university') or ''
        f['department']           = profile.get('department') or ''
        f['department_category']  = DEPARTMENT_CATEGORIES.get(f['department'], 'Other')
        f['degree']               = profile.get('degree') or 'Bachelor'
        f['degree_level']         = DEGREE_LEVELS.get(f['degree'], 1)
        f['year']                 = self._safe_float(profile.get('year'), 2) / 4.0
        f['is_professional']      = 1 if profile.get('is_professional') else 0
        f['work_schedule']        = profile.get('work_schedule') or 'Regular hours'
        f['work_from_home']       = self._safe_float(profile.get('work_from_home'), 0.3)
        f['ethnicity']            = profile.get('ethnicity') or 'N/A'
        f['home_city']            = profile.get('home_city') or 'N/A'
        f['home_region']          = self._get_region(f['home_city'])
        langs_raw                 = profile.get('languages') or 'Urdu, English'
        f['languages']            = [l.strip() for l in str(langs_raw).split(',')]
        f['study_habits']         = self._safe_float(profile.get('study_habits'), 0.5)
        raw_clean                 = profile.get('cleanliness_level')
        try:
            ck = int(float(raw_clean)) if raw_clean is not None else 3
        except (ValueError, TypeError):
            ck = 3
        f['cleanliness_level']    = CLEANLINESS_MAP.get(ck, 0.5)
        f['sleep_schedule']       = profile.get('sleep_schedule') or 'Flexible'
        f['social_preference']    = self._safe_float(profile.get('social_preference'), 0.5)
        f['noise_tolerance']      = self._safe_float(profile.get('noise_tolerance'), 0.5)
        f['personality']          = profile.get('personality') or 'Ambivert'
        f['food_preference']      = profile.get('food_preference') or 'Both'
        f['smoking_pref']         = profile.get('smoking_pref') or 'Non-smoker'
        f['guest_policy']         = profile.get('guest_policy') or 'Occasional guests'
        f['curfew_flexibility']   = self._safe_float(profile.get('curfew_flexibility'), 0.5)
        f['sharing_pref']         = profile.get('sharing_pref') or 'Flexible'
        f['gender']               = profile.get('gender') or 'Male'
        f['budget_min']           = self._safe_float(profile.get('budget_min'), 10000)
        f['budget_max']           = self._safe_float(profile.get('budget_max'), 20000)
        f['price_sensitivity']    = self._safe_float(profile.get('price_sensitivity'), 0.5)
        return f

    def calculate_pairwise_similarity(self, profile_a, profile_b):
        fa = self.extract_features(profile_a)
        fb = self.extract_features(profile_b)
        s  = {}

        # Academic
        s['university']   = 1.0 if fa['university'] == fb['university'] and fa['university'] else 0.0
        s['dept_cat']     = 1.0 if fa['department_category'] == fb['department_category'] else 0.0
        s['department']   = 1.0 if fa['department'] == fb['department'] and fa['department'] else 0.3
        s['degree_level'] = 1.0 - abs(fa['degree_level'] - fb['degree_level']) / 3.0
        s['year_diff']    = 1.0 - abs(fa['year'] - fb['year'])
        s['pro_compat']   = 1.0 if fa['is_professional'] == fb['is_professional'] else 0.7

        # Cultural
        s['ethnicity']    = 1.0 if fa['ethnicity'] == fb['ethnicity'] and fa['ethnicity'] != 'N/A' else 0.0
        s['region']       = 1.0 if fa['home_region'] == fb['home_region'] else 0.4
        s['hometown']     = 1.0 if fa['home_city'] == fb['home_city'] and fa['home_city'] != 'N/A' else 0.0
        s['language']     = self._language_sim(fa['languages'], fb['languages'])

        # Habits
        s['study']        = 1.0 - abs(fa['study_habits'] - fb['study_habits'])
        s['cleanliness']  = 1.0 - abs(fa['cleanliness_level'] - fb['cleanliness_level'])
        s['sleep']        = 1.0 if fa['sleep_schedule'] == fb['sleep_schedule'] else 0.0
        s['social']       = 1.0 - abs(fa['social_preference'] - fb['social_preference'])
        s['noise']        = 1.0 - abs(fa['noise_tolerance'] - fb['noise_tolerance'])
        s['personality']  = 1.0 if fa['personality'] == fb['personality'] else 0.5

        # Lifestyle
        s['food']         = 1.0 if fa['food_preference'] == fb['food_preference'] else 0.5
        s['smoking']      = 1.0 if fa['smoking_pref'] == fb['smoking_pref'] else 0.2
        s['guest']        = 1.0 if fa['guest_policy'] == fb['guest_policy'] else 0.5
        s['sharing']      = 1.0 if fa['sharing_pref'] == fb['sharing_pref'] else 0.6

        # Budget / gender
        s['gender']       = 1.0 if fa['gender'] == fb['gender'] else 0.4
        s['budget']       = self._budget_overlap(fa['budget_min'], fa['budget_max'],
                                                  fb['budget_min'], fb['budget_max'])

        # Simple weighted average (equal weight per category)
        total = (
            0.20 * (s['university'] + s['dept_cat'] + s['department'] + s['degree_level'] + s['year_diff']) / 5 +
            0.20 * (s['ethnicity'] + s['region'] + s['hometown'] + s['language']) / 4 +
            0.30 * (s['study'] + s['cleanliness'] + s['sleep'] + s['social'] + s['noise'] + s['personality']) / 6 +
            0.15 * (s['food'] + s['smoking'] + s['guest'] + s['sharing']) / 4 +
            0.15 * (s['gender'] + s['budget'] + s['pro_compat']) / 3
        )
        return round(min(1.0, max(0.0, total)), 3)

    def _language_sim(self, la, lb):
        if not la or not lb:
            return 0.5
        common = len(set(la) & set(lb))
        total  = len(set(la) | set(lb))
        base   = common / total if total else 0.5
        bonus  = 0.05 * ('English' in la and 'English' in lb) + 0.05 * ('Urdu' in la and 'Urdu' in lb)
        return min(1.0, base + bonus)

    def _budget_overlap(self, mna, mxa, mnb, mxb):
        ol_min = max(mna, mnb)
        ol_max = min(mxa, mxb)
        if ol_max <= ol_min:
            return 0.0
        ra = mxa - mna
        rb = mxb - mnb
        denom = min(ra, rb) if min(ra, rb) > 0 else 1
        return min(1.0, (ol_max - ol_min) / denom)