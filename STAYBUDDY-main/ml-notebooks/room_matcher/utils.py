# utils.py
"""
Utility functions for the room matching system
"""

import pandas as pd
from tabulate import tabulate

def display_match_result(match_result):
    """Pretty print a match result"""
    if not match_result:
        print("No match found")
        return
    
    print("\n" + "="*60)
    print(f"🤝 MATCH: {match_result['student_a']} ↔ {match_result['student_b']}")
    print("="*60)
    print(f"📊 Overall Match Score: {match_result['overall_score']*100:.1f}%")
    print(f"📝 Reason: {match_result['reasons']}")
    print("\n📈 Detailed Breakdown:")
    
    for key, value in match_result.get('breakdown', {}).items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        print(f"   {key.replace('_', ' ').title():15} {bar} {value*100:.0f}%")
    print("="*60 + "\n")

def display_room_recommendations(recommendations):
    """Pretty display room recommendations"""
    if not recommendations:
        print("No room recommendations found")
        return
    
    print("\n" + "="*70)
    print("🏠 ROOM RECOMMENDATIONS")
    print("="*70)
    
    table_data = []
    for room in recommendations:
        table_data.append([
            room['room_id'],
            room['room_type'],
            f"{room['current_occupants']}",
            room['vacancies'],
            f"{room['compatibility_score']*100:.0f}%",
            ", ".join(room['recommendation_reasons'][:2])
        ])
    
    headers = ['Room ID', 'Type', 'Current Occupants', 'Spots', 'Match', 'Key Reasons']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("="*70 + "\n")

def display_roommate_recommendations(roommates):
    """Pretty display roommate recommendations"""
    if not roommates:
        print("No roommate recommendations found")
        return
    
    print("\n" + "="*60)
    print("👥 POTENTIAL ROOMMATES")
    print("="*60)
    
    table_data = []
    for rm in roommates:
        table_data.append([
            rm['student_id'],
            f"{rm['similarity_score']*100:.0f}%",
            rm.get('department', 'N/A'),
            rm.get('university', 'N/A')[:15],
            rm.get('sleep_schedule', 'N/A'),
            f"Clean: {rm.get('cleanliness_level', 'N/A')}"
        ])
    
    headers = ['Student ID', 'Match', 'Department', 'University', 'Sleep', 'Cleanliness']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    print("="*60 + "\n")

def display_comprehensive_recommendation(comp_rec):
    """Display comprehensive recommendation"""
    if not comp_rec:
        return
    
    print("\n" + "="*70)
    print(f"📋 COMPREHENSIVE ROOM RECOMMENDATION FOR {comp_rec['student_id']}")
    print("="*70)
    
    print(f"\n📚 Student Profile:")
    print(f"   Department: {comp_rec['department']}")
    print(f"   Year: {comp_rec['year']}")
    
    print(f"\n⚙️ Your Preferences:")
    for key, value in comp_rec['preference_summary'].items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🏆 Top Room Recommendations:")
    display_room_recommendations(comp_rec['best_rooms'][:3])
    
    print(f"\n👥 Top Potential Roommates:")
    display_roommate_recommendations(comp_rec['best_roommates'])
    
    print("\n💡 Quick Actions:")
    if comp_rec.get('by_university'):
        print(f"   • {len(comp_rec['by_university'])} rooms with same university students available")
    if comp_rec.get('by_room_type'):
        print(f"   • {len(comp_rec['by_room_type'])} {comp_rec['preference_summary']['preferred_room_type']} rooms available")
    
    print("="*70 + "\n")

def save_recommendations_to_csv(recommendations, filename="room_recommendations.csv"):
    """Save recommendations to CSV"""
    if not recommendations:
        return
    
    df = pd.DataFrame(recommendations)
    df.to_csv(filename, index=False)
    print(f"✅ Recommendations saved to {filename}")