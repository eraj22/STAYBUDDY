# data_loader.py - FIXED VERSION
"""
Loads all_residents.csv which contains both students (STU-) and
job-seeking professionals (JOB-).  Correctly classifies each by
the ID prefix so every downstream module sees the right cohort.
"""

import pandas as pd
import numpy as np
import os
from config import MATCHES_CSV_PATH, COMBINED_CSV_PATH

STUDY_HABITS_TEXT_MAP = {
    'studious': 0.85,
    'moderate': 0.50,
    'social': 0.15,
}

def _to_float_study(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.5
    s = str(val).strip().lower()
    if s in STUDY_HABITS_TEXT_MAP:
        return STUDY_HABITS_TEXT_MAP[s]
    try:
        return float(s)
    except Exception:
        return 0.5


class DataLoader:
    def __init__(self):
        self.matches_df = None
        self.profiles_df = None
        self.all_profiles = {}
        self.student_profiles = {}
        self.professional_profiles = {}

    def load_data(self):
        try:
            if os.path.exists(MATCHES_CSV_PATH):
                self.matches_df = pd.read_csv(MATCHES_CSV_PATH)
                print(f"✅ Loaded {len(self.matches_df)} match records")
            else:
                print(f"⚠️  Match file not found: {MATCHES_CSV_PATH}")
                self.matches_df = pd.DataFrame(columns=['student_a','student_b','overall_score','reason'])

            combined_path = COMBINED_CSV_PATH if os.path.exists(COMBINED_CSV_PATH) else 'all_residents.csv'
            if not os.path.exists(combined_path):
                print(f"❌ Profile file not found: {combined_path}")
                return False

            self.profiles_df = pd.read_csv(combined_path)
            print(f"✅ Loaded {len(self.profiles_df)} resident rows")
            self._build_all_profiles()
            return True

        except Exception as exc:
            import traceback
            print(f"❌ Error loading data: {exc}")
            traceback.print_exc()
            return False

    def _build_all_profiles(self):
        self.all_profiles.clear()
        self.student_profiles.clear()
        self.professional_profiles.clear()

        for _, row in self.profiles_df.iterrows():
            rid = row.get('student_id') or row.get('user_id')
            if not rid:
                continue

            profile = {}
            for col, val in row.items():
                if isinstance(val, float) and np.isnan(val):
                    profile[col] = None
                else:
                    profile[col] = val

            # Normalise study_habits to float 0-1
            profile['study_habits'] = _to_float_study(profile.get('study_habits'))

            # Ensure key text fields have sensible defaults
            for field, default in [
                ('ethnicity',     'N/A'),
                ('home_city',     'N/A'),
                ('languages',     'Urdu, English'),
                ('personality',   'Ambivert'),
                ('smoking_pref',  'Non-smoker'),
                ('sleep_schedule','Flexible'),
                ('guest_policy',  'Occasional guests'),
                ('sharing_pref',  'Flexible'),
            ]:
                v = profile.get(field)
                if not v or str(v).strip() in ('', 'nan', 'None'):
                    profile[field] = default

            # Ensure numeric fields are float
            for field in ('social_preference', 'noise_tolerance', 'curfew_flexibility',
                          'price_sensitivity', 'comfort_preference'):
                try:
                    profile[field] = float(profile[field]) if profile.get(field) is not None else 0.5
                except (ValueError, TypeError):
                    profile[field] = 0.5

            for field in ('budget_min', 'budget_max'):
                try:
                    profile[field] = float(profile[field]) if profile.get(field) is not None else 10000
                except (ValueError, TypeError):
                    profile[field] = 10000

            rid_str = str(rid)
            self.all_profiles[rid_str] = profile

            if rid_str.startswith('JOB-'):
                profile['is_professional'] = True
                profile['resident_type'] = 'professional'
                self.professional_profiles[rid_str] = profile
            else:
                profile['is_professional'] = False
                profile['resident_type'] = 'student'
                self.student_profiles[rid_str] = profile

        print(f"\n📊 Profile counts:")
        print(f"   Total        : {len(self.all_profiles)}")
        print(f"   Students (STU-): {len(self.student_profiles)}")
        print(f"   Professionals (JOB-): {len(self.professional_profiles)}")

    def get_student_profile(self, resident_id):
        return self.all_profiles.get(str(resident_id))

    def get_all_students(self):
        return list(self.all_profiles.keys())

    def get_all_residents(self):
        return self.get_all_students()

    def get_students_only(self):
        return list(self.student_profiles.keys())

    def get_professionals_only(self):
        return list(self.professional_profiles.keys())

    def get_all_profiles_dict(self):
        return self.all_profiles