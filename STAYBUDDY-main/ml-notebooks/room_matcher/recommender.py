# recommender.py - FIXED VERSION

class RoomRecommender:
    def __init__(self, room_matcher):
        self.rm = room_matcher

    def recommend_rooms(self, resident_id, preferences=None):
        if preferences is None:
            preferences = {}

        profile = self.rm.dl.get_student_profile(resident_id)
        if not profile:
            return []

        from config import AVAILABLE_ROOMS
        filtered = {}
        for room_id, info in AVAILABLE_ROOMS.items():
            if len(info['current_occupants']) >= info['capacity']:
                continue

            if preferences.get('gender'):
                g = preferences['gender']
                if info['gender'].lower() not in (g.lower(), 'mixed'):
                    continue

            if preferences.get('room_type'):
                if info['type'].lower() != preferences['room_type'].lower():
                    continue

            if preferences.get('resident_type'):
                rt = preferences['resident_type'].lower()
                if info.get('resident_type', 'mixed').lower() not in (rt, 'mixed'):
                    continue

            filtered[room_id] = info

        rooms = self.rm.find_best_rooms(resident_id, filtered, top_k=10)

        for room in rooms:
            room['recommendation_reasons'] = self._reasons(room, profile, preferences)

        return rooms[:5]

    def _reasons(self, room, profile, preferences):
        reasons = list(room.get('reasons', []))
        if preferences.get('university'):
            reasons.append(f"🎓 {preferences['university']} filter applied")
        if preferences.get('ethnicity'):
            reasons.append(f"🌍 {preferences['ethnicity']} cultural preference")
        return reasons or [f"🛏️ {room['room_type']} room available"]

    def filter_by_university(self, resident_id, university):
        return self.recommend_rooms(resident_id, {'university': university})

    def filter_by_culture(self, resident_id, ethnicity):
        return self.recommend_rooms(resident_id, {'ethnicity': ethnicity})

    def filter_by_room_type(self, resident_id, room_type):
        return self.recommend_rooms(resident_id, {'room_type': room_type})

    def get_comprehensive_recommendation(self, resident_id):
        profile = self.rm.dl.get_student_profile(resident_id)
        if not profile:
            return None

        is_pro = bool(profile.get('is_professional'))
        study_h = float(profile.get('study_habits', 0.5))

        return {
            'student_id': resident_id,
            'is_professional': is_pro,
            'department': profile.get('department'),
            'year': profile.get('year'),
            'best_rooms': self.recommend_rooms(resident_id),
            'best_roommates': self.rm.find_best_roommates(resident_id, top_k=3),
            'by_room_type': self.filter_by_room_type(
                resident_id, profile.get('preferred_room_type', 'Double')
            ),
            'preference_summary': {
                'preferred_room_type': profile.get('preferred_room_type', 'Not specified'),
                'study_style': 'Studious' if study_h > 0.7 else 'Social' if study_h < 0.3 else 'Balanced',
                'sleep_schedule': profile.get('sleep_schedule', 'Not specified'),
                'cleanliness': str(profile.get('cleanliness_level', 3)) + '/5',
                'budget_range': f"PKR {int(profile.get('budget_min',0)):,} – {int(profile.get('budget_max',0)):,}",
                'resident_type': 'Professional' if is_pro else 'Student',
            }
        }