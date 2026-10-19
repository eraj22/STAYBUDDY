# check_data.py
import pandas as pd

# Load the merged file
df = pd.read_csv('all_residents.csv')

print("="*60)
print("CHECKING ALL_RESIDENTS.CSV")
print("="*60)

print(f"\nTotal rows: {len(df)}")
print(f"Columns: {list(df.columns)[:15]}...")

# Check ID columns
if 'student_id' in df.columns:
    print(f"\n✅ student_id column found")
    print(f"   Sample student_ids: {df['student_id'].head(10).tolist()}")
    
    # Count by prefix
    stu_count = df['student_id'].str.startswith('STU-', na=False).sum()
    job_count = df['student_id'].str.startswith('JOB-', na=False).sum()
    
    print(f"\n📊 By ID prefix:")
    print(f"   STU- (Students): {stu_count}")
    print(f"   JOB- (Professionals): {job_count}")

# Check for professional indicators
if 'job_title' in df.columns:
    has_job = df['job_title'].notna().sum()
    print(f"\n📊 Has job_title: {has_job}")

if 'is_professional' in df.columns:
    is_pro = df['is_professional'].sum()
    print(f"   is_professional = True: {is_pro}")

if 'university' in df.columns:
    working_pro = (df['university'] == 'Working Professional').sum()
    print(f"   University = 'Working Professional': {working_pro}")

print("\n" + "="*60)