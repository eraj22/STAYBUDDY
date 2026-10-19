# merge_data.py
"""
Merge student profiles with job seekers/professionals data
Run this script once to create the combined dataset
"""

import pandas as pd
import numpy as np
import os

def load_job_seekers(file_path='job_seekers.csv'):
    """Load job seekers data with multiple encoding attempts"""
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"✅ Loaded job_seekers.csv with encoding: {encoding}")
            print(f"   Records: {len(df)}")
            print(f"   Columns: {list(df.columns)[:10]}...")
            return df
        except Exception as e:
            continue
    
    print("❌ Could not load job_seekers.csv")
    print("Please check:")
    print("1. File exists in the current directory")
    print("2. File name is exactly 'job_seekers.csv'")
    print(f"3. Current directory: {os.getcwd()}")
    
    # List files in current directory
    print("\n📁 Files in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.csv'):
            print(f"   - {file}")
    
    return None

def transform_job_seeker_to_student_format(job_df):
    """Transform job seeker dataframe to match student profile format"""
    
    print("\n🔄 Transforming job seeker data...")
    
    # Create new dataframe with student profile structure
    transformed = pd.DataFrame()
    
    # Map columns
    column_mapping = {
        'user_id': 'student_id',
        'gender': 'gender',
        'age': 'year',  # Age as year (will be normalized)
        'occupation': 'job_title',
        'workplace_name': 'company',
        'workplace_lat': 'latitude',
        'workplace_lon': 'longitude',
        'budget_min': 'budget_min',
        'budget_max': 'budget_max',
        'price_sensitivity': 'price_sensitivity',
        'max_distance_km': 'max_distance_km',
        'needs_transport': 'needs_transport',
        'preferred_room_type': 'preferred_room_type',
        'food_preference': 'food_preference',
        'work_env_preference': 'study_habits',
        'social_preference': 'social_preference',
        'comfort_preference': 'comfort_preference',
        'noise_tolerance': 'noise_tolerance',
        'curfew_flexibility': 'curfew_flexibility',
        'must_have_amenities': 'must_have_amenities',
        'priority_wifi': 'priority_wifi',
        'priority_parking': 'priority_parking',
        'priority_laundry': 'priority_laundry',
        'priority_power_backup': 'priority_generator',
        'priority_ac': 'priority_ac',
        'priority_hot_water': 'priority_hot_water',
        'priority_security': 'priority_security_guard',
        'priority_prayer_room': 'priority_prayer_room'
    }
    
    # Apply mapping for existing columns
    for old_col, new_col in column_mapping.items():
        if old_col in job_df.columns:
            transformed[new_col] = job_df[old_col]
        else:
            print(f"⚠️ Column not found: {old_col}")
    
    # Add default values for missing columns
    transformed['department'] = job_df['occupation'].apply(lambda x: x if pd.notna(x) else 'Professional')
    transformed['degree'] = 'Professional'
    transformed['university'] = 'Working Professional'
    transformed['ethnicity'] = 'N/A'
    transformed['home_city'] = 'Islamabad'  # Default based on workplace area
    transformed['languages'] = 'Urdu, English'
    transformed['personality'] = 'Ambivert'
    transformed['smoking_pref'] = 'Non-smoker'
    transformed['cleanliness_level'] = np.random.randint(3, 6, len(job_df))
    transformed['sleep_schedule'] = job_df['curfew_flexibility'].apply(
        lambda x: 'Night owl' if x > 0.6 else 'Early bird' if x < 0.4 else 'Flexible'
    )
    transformed['guest_policy'] = 'Occasional guests'
    transformed['sharing_pref'] = 'Flexible'
    transformed['work_schedule'] = 'Regular hours'
    transformed['work_from_home'] = 0.3
    transformed['years_experience'] = np.random.randint(1, 10, len(job_df))
    
    # Set year (age converted to academic year equivalent)
    transformed['year'] = job_df['age'].apply(lambda x: min(4, max(1, int(x/5))) if pd.notna(x) else 2)
    
    # Set study habits (work environment preference)
    if 'study_habits' in transformed.columns:
        transformed['study_habits'] = transformed['study_habits']
    else:
        transformed['study_habits'] = 0.5
    
    # Add is_professional flag (will be used to identify)
    transformed['is_professional'] = True
    
    # Add job title as department for filtering
    transformed['department'] = job_df['occupation'].fillna('Professional')
    
    print(f"✅ Transformed {len(transformed)} job seeker records")
    return transformed

def merge_all_data(student_csv='student_profiles_enhanced.csv', job_csv='job_seekers.csv', output_csv='all_residents.csv'):
    """Merge student and job seeker data into one file"""
    
    print("\n" + "="*60)
    print("🔄 MERGING STUDENT AND JOB SEEKER DATA")
    print("="*60)
    
    # Load student data
    try:
        students_df = pd.read_csv(student_csv)
        print(f"✅ Loaded {len(students_df)} student records")
    except Exception as e:
        print(f"❌ Error loading student data: {e}")
        return None
    
    # Load job seeker data
    job_df = load_job_seekers(job_csv)
    if job_df is None:
        print("\n⚠️ Job seeker data not found. Using only student data.")
        # Save student data only
        students_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(students_df)} records to {output_csv}")
        return students_df
    
    # Transform job seeker data
    job_transformed = transform_job_seeker_to_student_format(job_df)
    
    # Combine datasets
    combined_df = pd.concat([students_df, job_transformed], ignore_index=True)
    
    # Add resident_type column
    combined_df['resident_type'] = combined_df.apply(
        lambda x: 'professional' if x.get('is_professional', False) else 'student', 
        axis=1
    )
    
    # Save combined data
    combined_df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("✅ MERGE COMPLETE")
    print("="*60)
    print(f"📊 Total residents: {len(combined_df)}")
    print(f"   - Students: {len(students_df)}")
    print(f"   - Professionals: {len(job_transformed)}")
    print(f"💾 Saved to: {output_csv}")
    
    # Print sample of professionals
    if len(job_transformed) > 0:
        print("\n📋 Sample of professional records:")
        sample_cols = ['student_id', 'job_title', 'department', 'budget_min', 'budget_max']
        available_cols = [c for c in sample_cols if c in job_transformed.columns]
        print(job_transformed[available_cols].head(10))
    
    return combined_df

def update_available_rooms_for_professionals():
    """Update available rooms to include more professional rooms"""
    
    from config import AVAILABLE_ROOMS
    
    # Add more professional rooms based on job seeker count
    additional_rooms = {
        'Room-P06': {'capacity': 2, 'current_occupants': [], 'type': 'Double', 'gender': 'Mixed', 'building': 'P', 'floor': 3, 'resident_type': 'professionals'},
        'Room-P07': {'capacity': 1, 'current_occupants': [], 'type': 'Single', 'gender': 'Boys', 'building': 'P', 'floor': 3, 'resident_type': 'professionals'},
        'Room-P08': {'capacity': 1, 'current_occupants': [], 'type': 'Single', 'gender': 'Girls', 'building': 'P', 'floor': 3, 'resident_type': 'professionals'},
        'Room-P09': {'capacity': 4, 'current_occupants': [], 'type': 'Dormitory', 'gender': 'Mixed', 'building': 'P', 'floor': 4, 'resident_type': 'professionals'},
        'Room-M05': {'capacity': 2, 'current_occupants': [], 'type': 'Double', 'gender': 'Mixed', 'building': 'M', 'floor': 3, 'resident_type': 'mixed'},
        'Room-M06': {'capacity': 2, 'current_occupants': [], 'type': 'Double', 'gender': 'Mixed', 'building': 'M', 'floor': 3, 'resident_type': 'mixed'},
    }
    
    AVAILABLE_ROOMS.update(additional_rooms)
    print(f"✅ Added {len(additional_rooms)} new rooms")
    print(f"📊 Total rooms now: {len(AVAILABLE_ROOMS)}")
    
    return AVAILABLE_ROOMS

if __name__ == "__main__":
    # Run the merge
    combined = merge_all_data()
    
    if combined is not None:
        print("\n" + "="*60)
        print("🎯 NEXT STEPS:")
        print("="*60)
        print("1. Update data_loader.py to load 'all_residents.csv' instead of separate files")
        print("2. Restart the web application")
        print("3. Both students and professionals will be available in the system")