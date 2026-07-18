"""
rebuild_data.py
Run once to fix all_residents.csv:
  - Sets correct resident_type for STU vs JOB rows
  - Normalises study_habits to numeric (0-1) for all rows
"""

import pandas as pd
import numpy as np

STUDY_HABITS_TEXT_MAP = {
    'studious': 0.85,
    'moderate': 0.50,
    'social': 0.15,
}

def normalize_study_habits(val):
    if pd.isna(val):
        return 0.5
    s = str(val).strip().lower()
    if s in STUDY_HABITS_TEXT_MAP:
        return STUDY_HABITS_TEXT_MAP[s]
    try:
        return float(s)
    except:
        return 0.5

df = pd.read_csv('all_residents.csv')
print(f"Loaded {len(df)} rows")

# Fix resident_type
df['resident_type'] = df['student_id'].apply(
    lambda x: 'professional' if str(x).startswith('JOB-') else 'student'
)

# Fix is_professional
df['is_professional'] = df['student_id'].apply(
    lambda x: True if str(x).startswith('JOB-') else False
)

# Fix study_habits → always numeric float
df['study_habits'] = df['study_habits'].apply(normalize_study_habits)

# For JOB- rows that have no ethnicity, default to empty string not NaN
df['ethnicity'] = df['ethnicity'].fillna('N/A')
df['home_city'] = df['home_city'].fillna('N/A')
df['languages'] = df['languages'].fillna('Urdu, English')
df['personality'] = df['personality'].fillna('Ambivert')
df['smoking_pref'] = df['smoking_pref'].fillna('Non-smoker')
df['sleep_schedule'] = df['sleep_schedule'].fillna('Flexible')
df['guest_policy'] = df['guest_policy'].fillna('Occasional guests')
df['sharing_pref'] = df['sharing_pref'].fillna('Flexible')

df.to_csv('all_residents.csv', index=False)
print(f"Fixed all_residents.csv")
print(f"  Students (STU-): {(df['resident_type']=='student').sum()}")
print(f"  Professionals (JOB-): {(df['resident_type']=='professional').sum()}")
print(f"  study_habits sample: {df['study_habits'].head(5).tolist()}")

if __name__ == '__main__':
    pass