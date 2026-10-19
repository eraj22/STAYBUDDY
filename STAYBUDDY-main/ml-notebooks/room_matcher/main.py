# main.py
"""
AI Room Matching System - Main Application
"""

import sys
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from similarity_scorer import SimilarityScorer
from room_matcher import RoomMatcher
from recommender import RoomRecommender
from utils import *

def main():
    print("\n" + "="*60)
    print("🏠 AI ROOM MATCHING SYSTEM")
    print("="*60)
    print("Welcome! This system helps you find the perfect roommate and room.")
    print("-"*60)
    
    # Initialize components
    print("\n🔧 Initializing system...")
    data_loader = DataLoader()
    if not data_loader.load_data():
        print("❌ Failed to load data. Exiting...")
        return
    
    feature_engineer = FeatureEngineer()
    similarity_scorer = SimilarityScorer(feature_engineer)
    
    # Try to load trained model, or train new one
    if similarity_scorer.load_model():
        print("✅ Using pre-trained model")
    else:
        print("📊 Training new similarity model...")
        X, y = similarity_scorer.prepare_training_data(
            data_loader.matches_df, 
            data_loader.student_profiles
        )
        similarity_scorer.train(X, y)
        similarity_scorer.save_model()
    
    room_matcher = RoomMatcher(data_loader, feature_engineer, similarity_scorer)
    recommender = RoomRecommender(room_matcher)
    
    # Get available students
    students = data_loader.get_all_students()
    print(f"\n📋 Loaded {len(students)} student profiles")
    
    while True:
        print("\n" + "-"*50)
        print("MAIN MENU")
        print("-"*50)
        print("1. 🎯 Find Best Roommates for a Student")
        print("2. 🏠 Find Best Rooms for a Student")
        print("3. 🔍 Filter Rooms by Preferences")
        print("4. 📊 Get Comprehensive Recommendation")
        print("5. 💑 Check Compatibility Between Two Students")
        print("6. 🌍 Filter by University")
        print("7. 🎭 Filter by Cultural Background")
        print("8. 📈 View Student Profile")
        print("9. ❌ Exit")
        print("-"*50)
        
        choice = input("\nEnter your choice (1-9): ").strip()
        
        if choice == '1':
            student_id = input("Enter Student ID (e.g., STU-001): ").strip().upper()
            if student_id in data_loader.student_profiles:
                roommates = room_matcher.find_best_roommates(student_id, top_k=5)
                display_roommate_recommendations(roommates)
            else:
                print(f"❌ Student {student_id} not found")
        
        elif choice == '2':
            student_id = input("Enter Student ID: ").strip().upper()
            if student_id in data_loader.student_profiles:
                from config import AVAILABLE_ROOMS
                rooms = room_matcher.find_best_rooms(student_id, AVAILABLE_ROOMS, top_k=5)
                display_room_recommendations(rooms)
            else:
                print(f"❌ Student {student_id} not found")
        
        elif choice == '3':
            student_id = input("Enter Student ID: ").strip().upper()
            if student_id in data_loader.student_profiles:
                print("\nFilter options:")
                print("1. Room Type (Single/Double/Dormitory)")
                print("2. Gender (Boys/Girls/Mixed)")
                print("3. University")
                filter_choice = input("Choose filter (1-3): ")
                
                preferences = {}
                if filter_choice == '1':
                    room_type = input("Enter room type (Single/Double/Dormitory): ").strip()
                    preferences['room_type'] = room_type
                elif filter_choice == '2':
                    gender = input("Enter gender (Boys/Girls/Mixed): ").strip()
                    preferences['gender'] = gender
                elif filter_choice == '3':
                    university = input("Enter university name: ").strip()
                    preferences['university'] = university
                
                rooms = recommender.recommend_rooms(student_id, preferences)
                display_room_recommendations(rooms)
            else:
                print(f"❌ Student {student_id} not found")
        
        elif choice == '4':
            student_id = input("Enter Student ID: ").strip().upper()
            if student_id in data_loader.student_profiles:
                comp_rec = recommender.get_comprehensive_recommendation(student_id)
                display_comprehensive_recommendation(comp_rec)
                
                save = input("Save recommendations to CSV? (y/n): ").strip().lower()
                if save == 'y':
                    save_recommendations_to_csv(comp_rec.get('best_rooms', []))
            else:
                print(f"❌ Student {student_id} not found")
        
        elif choice == '5':
            student_a = input("Enter first Student ID: ").strip().upper()
            student_b = input("Enter second Student ID: ").strip().upper()
            if student_a in data_loader.student_profiles and student_b in data_loader.student_profiles:
                match = room_matcher.match_pair(student_a, student_b)
                display_match_result(match)
            else:
                print("❌ One or both students not found")
        
        elif choice == '6':
            student_id = input("Enter your Student ID: ").strip().upper()
            if student_id in data_loader.student_profiles:
                profile = data_loader.get_student_profile(student_id)
                uni = profile.get('university')
                print(f"🎓 Finding rooms with {uni} students...")
                rooms = recommender.filter_by_university(student_id, uni)
                display_room_recommendations(rooms)
            else:
                print(f"❌ Student {student_id} not found")
        
        elif choice == '7':
            student_id = input("Enter your Student ID: ").strip().upper()
            if student_id in data_loader.student_profiles:
                profile = data_loader.get_student_profile(student_id)
                ethnicity = profile.get('ethnicity')
                print(f"🌍 Finding rooms with {ethnicity} students...")
                rooms = recommender.filter_by_culture(student_id, ethnicity)
                display_room_recommendations(rooms)
            else:
                print(f"❌ Student {student_id} not found")
        
        elif choice == '8':
            student_id = input("Enter Student ID: ").strip().upper()
            profile = data_loader.get_student_profile(student_id)
            if profile:
                print("\n" + "="*50)
                print(f"📋 STUDENT PROFILE: {student_id}")
                print("="*50)
                for key, value in profile.items():
                    if key not in ['latitude', 'longitude', 'must_have_amenities']:
                        print(f"   {key.replace('_', ' ').title()}: {value}")
                print("="*50)
            else:
                print(f"❌ Student {student_id} not found")
        
        elif choice == '9':
            print("\n👋 Thank you for using AI Room Matching System!")
            print("Good luck finding your perfect roommate! 🏠")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
