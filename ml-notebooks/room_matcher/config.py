# config.py - FIXED VERSION (no Mixed gender rooms)

import os

MATCHES_CSV_PATH  = "roommate_matches.csv"
PROFILES_CSV_PATH = "student_profiles_enhanced.csv"
COMBINED_CSV_PATH = "all_residents.csv"

FEATURE_WEIGHTS = {
    'university_match':       0.10,
    'department_match':       0.08,
    'degree_match':           0.07,
    'year_level':             0.04,
    'work_schedule':          0.06,
    'ethnicity_match':        0.08,
    'hometown_similarity':    0.06,
    'languages_shared':       0.05,
    'religious_considerations': 0.04,
    'study_habits':           0.08,
    'cleanliness_level':      0.07,
    'sleep_schedule':         0.06,
    'social_preference':      0.05,
    'noise_tolerance':        0.05,
    'guest_policy':           0.04,
    'food_preference':        0.05,
    'smoking_pref':           0.04,
    'curfew_flexibility':     0.03,
    'sharing_pref':           0.03,
    'work_from_home':         0.03,
    'work_hours':             0.02,
}

GENDER_MAP         = {'Female': 0, 'Male': 1, 'Other': 2}
SMOKING_MAP        = {'Non-smoker': 0, "Don't mind": 1, 'Smoker': 2}
SLEEP_SCHEDULE_MAP = {'Early bird': 0, 'Night owl': 1, 'Flexible': 2}
CLEANLINESS_MAP    = {1: 0.0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1.0}
FOOD_PREF_MAP      = {'Veg': 0, 'Non-Veg': 1, 'Both': 2}
GUEST_POLICY_MAP   = {'No guests': 0, 'Occasional guests': 1, 'Frequent guests': 2}
SHARING_PREF_MAP   = {'Prefers single': 0, 'Prefers sharing': 1, 'Flexible': 2}
PERSONALITY_MAP    = {'Introvert': 0, 'Ambivert': 1, 'Extrovert': 2}
WORK_SCHEDULE_MAP  = {'Regular hours': 0, 'Flexible': 1, 'Night shift': 2, 'Remote': 3}

DEPARTMENT_CATEGORIES = {
    'Computer Science':      'Tech',
    'Software Engineering':  'Tech',
    'Data Science':          'Tech',
    'Data Analyst':          'Tech',
    'Cyber Security':        'Tech',
    'IT':                    'Tech',
    'Electrical Engineering':'Engineering',
    'Civil Engineering':     'Engineering',
    'Mechanical Engineering':'Engineering',
    'BBA':                   'Business',
    'MBA':                   'Business',
    'Marketing':             'Business',
    'Finance':               'Business',
    'Accounting':            'Business',
    'Social Sciences':       'Humanities',
    'Psychology':            'Humanities',
    'Medicine':              'Medical',
    'Pharmacy':              'Medical',
    'Law':                   'Legal',
}

DEGREE_LEVELS = {
    'Bachelor': 1, 'BS': 1, 'BBA': 1, 'B.Sc': 1,
    'Master': 2,   'MS': 2, 'MBA': 2, 'M.Sc': 2,
    'PhD': 3,      'Doctorate': 3,
    'Professional': 2,
    'Computer Science': 1,
}

REGIONAL_CLUSTERS = {
    'North':   ['Islamabad', 'Rawalpindi', 'Abbottabad', 'Gilgit', 'Skardu', 'Hunza', 'Nagar'],
    'Central': ['Lahore', 'Faisalabad', 'Gujranwala', 'Multan', 'Sahiwal'],
    'South':   ['Karachi', 'Hyderabad', 'Sukkur', 'Mirpur Khas', 'Larkana'],
    'West':    ['Peshawar', 'Kohat', 'Mardan', 'Swat', 'Quetta', 'Zhob', 'Turbat'],
}

# ── Available rooms: Boys and Girls only — no Mixed gender ─────────────────
AVAILABLE_ROOMS = {
    # ── Student Building A ──────────────────────────────────────────────────
    'Room-A01': {'capacity': 2, 'current_occupants': ['STU-002'], 'type': 'Double',    'gender': 'Boys',  'building': 'A', 'floor': 1, 'resident_type': 'students'},
    'Room-A02': {'capacity': 2, 'current_occupants': ['STU-005'], 'type': 'Double',    'gender': 'Girls', 'building': 'A', 'floor': 1, 'resident_type': 'students'},
    'Room-A03': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Boys',  'building': 'A', 'floor': 2, 'resident_type': 'students'},
    'Room-A04': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Girls', 'building': 'A', 'floor': 2, 'resident_type': 'students'},

    # ── Student Building B ──────────────────────────────────────────────────
    'Room-B01': {'capacity': 4, 'current_occupants': ['STU-001', 'STU-003'], 'type': 'Dormitory', 'gender': 'Girls', 'building': 'B', 'floor': 1, 'resident_type': 'students'},
    'Room-B02': {'capacity': 2, 'current_occupants': [],          'type': 'Double',    'gender': 'Boys',  'building': 'B', 'floor': 2, 'resident_type': 'students'},
    'Room-B03': {'capacity': 2, 'current_occupants': [],          'type': 'Double',    'gender': 'Girls', 'building': 'B', 'floor': 2, 'resident_type': 'students'},
    'Room-B04': {'capacity': 4, 'current_occupants': [],          'type': 'Dormitory', 'gender': 'Boys',  'building': 'B', 'floor': 3, 'resident_type': 'students'},

    # ── Student Building C ──────────────────────────────────────────────────
    'Room-C01': {'capacity': 4, 'current_occupants': [],          'type': 'Dormitory', 'gender': 'Boys',  'building': 'C', 'floor': 1, 'resident_type': 'students'},
    'Room-C02': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Girls', 'building': 'C', 'floor': 1, 'resident_type': 'students'},
    'Room-C03': {'capacity': 2, 'current_occupants': [],          'type': 'Double',    'gender': 'Boys',  'building': 'C', 'floor': 2, 'resident_type': 'students'},
    'Room-C04': {'capacity': 2, 'current_occupants': [],          'type': 'Double',    'gender': 'Girls', 'building': 'C', 'floor': 2, 'resident_type': 'students'},

    # ── Professional Building P ─────────────────────────────────────────────
    'Room-P01': {'capacity': 2, 'current_occupants': [],          'type': 'Double',    'gender': 'Boys',  'building': 'P', 'floor': 1, 'resident_type': 'professionals'},
    'Room-P02': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Girls', 'building': 'P', 'floor': 1, 'resident_type': 'professionals'},
    'Room-P03': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Boys',  'building': 'P', 'floor': 2, 'resident_type': 'professionals'},
    'Room-P04': {'capacity': 2, 'current_occupants': [],          'type': 'Double',    'gender': 'Girls', 'building': 'P', 'floor': 2, 'resident_type': 'professionals'},
    'Room-P05': {'capacity': 4, 'current_occupants': [],          'type': 'Dormitory', 'gender': 'Boys',  'building': 'P', 'floor': 3, 'resident_type': 'professionals'},
    'Room-P06': {'capacity': 4, 'current_occupants': [],          'type': 'Dormitory', 'gender': 'Girls', 'building': 'P', 'floor': 3, 'resident_type': 'professionals'},
    'Room-P07': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Boys',  'building': 'P', 'floor': 4, 'resident_type': 'professionals'},
    'Room-P08': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Girls', 'building': 'P', 'floor': 4, 'resident_type': 'professionals'},

    # ── Mixed Building M (Students + Professionals, gender-separated) ───────
    'Room-M01': {'capacity': 2, 'current_occupants': [],          'type': 'Double',    'gender': 'Boys',  'building': 'M', 'floor': 1, 'resident_type': 'mixed'},
    'Room-M02': {'capacity': 2, 'current_occupants': ['STU-050'], 'type': 'Double',    'gender': 'Girls', 'building': 'M', 'floor': 1, 'resident_type': 'mixed'},
    'Room-M03': {'capacity': 4, 'current_occupants': [],          'type': 'Dormitory', 'gender': 'Boys',  'building': 'M', 'floor': 2, 'resident_type': 'mixed'},
    'Room-M04': {'capacity': 4, 'current_occupants': [],          'type': 'Dormitory', 'gender': 'Girls', 'building': 'M', 'floor': 2, 'resident_type': 'mixed'},
    'Room-M05': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Boys',  'building': 'M', 'floor': 3, 'resident_type': 'mixed'},
    'Room-M06': {'capacity': 1, 'current_occupants': [],          'type': 'Single',    'gender': 'Girls', 'building': 'M', 'floor': 3, 'resident_type': 'mixed'},
}

DEFAULT_TOP_K = 5
MIN_SIMILARITY_THRESHOLD = 0.5