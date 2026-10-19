# add_professionals.py - Add working professionals to the system
import pandas as pd
import random

# Read existing profiles
df = pd.read_csv('student_profiles_enhanced.csv')

# Add some working professionals
professionals = [
    {
        'student_id': 'PRO-001', 'gender': 'Male', 'department': 'Software Engineering',
        'year': 0, 'latitude': 33.72, 'longitude': 73.05, 'budget_min': 20000,
        'budget_max': 35000, 'price_sensitivity': 0.3, 'max_distance_km': 5,
        'needs_transport': 1, 'preferred_type': 'Boys', 'preferred_room_type': 'Single',
        'food_preference': 'Non-Veg', 'study_preference': 0.7, 'social_preference': 0.3,
        'comfort_preference': 0.6, 'noise_tolerance': 0.8, 'curfew_flexibility': 0.9,
        'must_have_amenities': '["WiFi", "AC"]', 'priority_wifi': 1, 'priority_gym': 0,
        'priority_study_room': 1, 'priority_cafeteria': 0, 'priority_laundry': 1,
        'priority_ac': 1, 'priority_hot_water': 1, 'priority_generator': 0,
        'ethnicity': 'Punjabi', 'home_city': 'Lahore', 'languages': 'Urdu, English, Punjabi',
        'university': 'Working Professional', 'degree': 'Professional', 'degree_level': 'Professional',
        'smoking_pref': 'Non-smoker', 'cleanliness_level': 4, 'sleep_schedule': 'Early bird',
        'guest_policy': 'Occasional guests', 'sharing_pref': 'Prefers single',
        'personality': 'Introvert', 'study_habits': 0.8, 'job_title': 'Software Engineer',
        'company': 'Tech Corp', 'work_schedule': 'Regular hours', 'work_from_home': 0.3
    },
    {
        'student_id': 'PRO-002', 'gender': 'Female', 'department': 'Marketing',
        'year': 0, 'latitude': 33.71, 'longitude': 73.04, 'budget_min': 18000,
        'budget_max': 28000, 'price_sensitivity': 0.4, 'max_distance_km': 4,
        'needs_transport': 1, 'preferred_type': 'Girls', 'preferred_room_type': 'Double',
        'food_preference': 'Both', 'study_preference': 0.4, 'social_preference': 0.7,
        'comfort_preference': 0.8, 'noise_tolerance': 0.5, 'curfew_flexibility': 0.7,
        'must_have_amenities': '["Laundry", "Gym"]', 'priority_wifi': 1, 'priority_gym': 1,
        'priority_study_room': 0, 'priority_cafeteria': 1, 'priority_laundry': 1,
        'priority_ac': 0, 'priority_hot_water': 1, 'priority_generator': 0,
        'ethnicity': 'Pathan', 'home_city': 'Peshawar', 'languages': 'Urdu, English, Pashto',
        'university': 'Working Professional', 'degree': 'Professional', 'degree_level': 'Professional',
        'smoking_pref': 'Non-smoker', 'cleanliness_level': 3, 'sleep_schedule': 'Night owl',
        'guest_policy': 'Frequent guests', 'sharing_pref': 'Prefers sharing',
        'personality': 'Extrovert', 'study_habits': 0.3, 'job_title': 'Marketing Manager',
        'company': 'AdAgency', 'work_schedule': 'Flexible', 'work_from_home': 0.5
    }
]

# Add professionals to dataframe
df_new = pd.concat([df, pd.DataFrame(professionals)], ignore_index=True)
df_new.to_csv('student_profiles_enhanced_with_pros.csv', index=False)
print(f"✅ Added professionals. Total profiles: {len(df_new)}")