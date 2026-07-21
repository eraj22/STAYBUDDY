# test_api.py - FIXED DISPLAY
from data_loader import DataLoader

print("Testing DataLoader...")
dl = DataLoader()
if dl.load_data():
    print("\n✅ Data loaded successfully!")
    
    all_residents = dl.get_all_residents()
    print(f"\n📊 Total residents: {len(all_residents)}")
    
    print(f"\n📋 First 10 residents:")
    for rid in all_residents[:10]:
        profile = dl.get_student_profile(rid)
        # Fix: Check by ID prefix
        if str(rid).startswith('JOB-'):
            is_pro = True
        elif str(rid).startswith('STU-'):
            is_pro = False
        else:
            is_pro = bool(profile.get('job_title')) or profile.get('is_professional') == True
        
        pro_tag = "👔" if is_pro else "🎓"
        dept = profile.get('department', 'N/A')[:25]
        title = profile.get('job_title', 'Student') if is_pro else 'Student'
        print(f"   {pro_tag} {rid}: {dept} - {title}")
    
    print(f"\n📊 Students only: {len(dl.get_students_only())}")
    print(f"📊 Professionals only: {len(dl.get_professionals_only())}")
else:
    print("❌ Failed to load data")