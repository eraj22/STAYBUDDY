# room_matcher.py - FIXED VERSION

import numpy as np
import pandas as pd
from config import *


class RoomMatcher:
    def __init__(self, data_loader, feature_engineer, similarity_scorer):
        self.dl = data_loader
        self.fe = feature_engineer
        self.ss = similarity_scorer

    # ------------------------------------------------------------------
    def find_best_roommates(self, resident_id, top_k=5, exclude_ids=None,
                             search_pool='all'):
        """
        Find compatible roommates for any resident (student OR professional).

        search_pool: 'all' | 'students' | 'professionals'
        """
        my_profile = self.dl.get_student_profile(resident_id)
        if not my_profile:
            return []

        exclude = {str(resident_id)}
        if exclude_ids:
            exclude.update([str(x) for x in exclude_ids])

        # Choose pool
        if search_pool == 'students':
            pool = self.dl.student_profiles
        elif search_pool == 'professionals':
            pool = self.dl.professional_profiles
        else:
            pool = self.dl.all_profiles   # ← KEY FIX: search everyone

        candidates = []
        for other_id, other_profile in pool.items():
            if other_id in exclude:
                continue
            similarity = self.ss.predict_similarity(my_profile, other_profile)
            details    = self._match_details(my_profile, other_profile)

            study_h = self._safe_float(other_profile.get('study_habits'), 0.5)
            clean   = other_profile.get('cleanliness_level', 3)
            try:
                clean = int(float(clean))
            except:
                clean = 3

            candidates.append({
                'student_id':        other_id,
                'similarity_score':  similarity,
                'match_details':     details,
                'gender':            other_profile.get('gender', 'N/A'),
                'department':        other_profile.get('department', 'N/A'),
                'degree':            other_profile.get('degree', 'N/A'),
                'year':              other_profile.get('year', 'N/A'),
                'university':        other_profile.get('university', 'N/A'),
                'ethnicity':         other_profile.get('ethnicity', 'N/A'),
                'home_city':         other_profile.get('home_city', 'N/A'),
                'personality':       other_profile.get('personality', 'N/A'),
                'study_habits':      study_h,
                'sleep_schedule':    other_profile.get('sleep_schedule', 'N/A'),
                'cleanliness_level': clean,
                'food_preference':   other_profile.get('food_preference', 'N/A'),
                'social_preference': other_profile.get('social_preference', 'N/A'),
                'noise_tolerance':   other_profile.get('noise_tolerance', 'N/A'),
                'smoking_pref':      other_profile.get('smoking_pref', 'N/A'),
                'guest_policy':      other_profile.get('guest_policy', 'N/A'),
                'is_professional':   bool(other_profile.get('is_professional')),
                'job_title':         other_profile.get('job_title') or 'Student',
            })

        candidates.sort(key=lambda x: -x['similarity_score'])
        return candidates[:top_k]

    # ------------------------------------------------------------------
    def find_best_rooms(self, resident_id, available_rooms, top_k=5, filters=None):
        my_profile = self.dl.get_student_profile(resident_id)
        if not my_profile:
            return []

        room_scores = []
        for room_id, room_info in available_rooms.items():
            if len(room_info['current_occupants']) >= room_info['capacity']:
                continue

            if room_info['current_occupants']:
                sims, rm_details = [], []
                for occ_id in room_info['current_occupants']:
                    occ_profile = self.dl.get_student_profile(occ_id)
                    if occ_profile:
                        sim = self.ss.predict_similarity(my_profile, occ_profile)
                        sims.append(sim)
                        rm_details.append({
                            'id': occ_id,
                            'similarity': sim,
                            'department': occ_profile.get('department', 'N/A'),
                            'ethnicity':  occ_profile.get('ethnicity', 'N/A'),
                            'personality':occ_profile.get('personality', 'N/A'),
                            'sleep_schedule': occ_profile.get('sleep_schedule', 'N/A'),
                            'cleanliness':    occ_profile.get('cleanliness_level', 'N/A'),
                            'is_professional':bool(occ_profile.get('is_professional')),
                        })
                avg_sim = float(np.mean(sims)) if sims else 0.5
                reasons = self._room_reasons(my_profile, rm_details)
            else:
                avg_sim    = 1.0
                rm_details = []
                reasons    = ["✨ Empty room – choose your own roommate!"]

            room_scores.append({
                'room_id':                room_id,
                'room_type':              room_info['type'],
                'capacity':               room_info['capacity'],
                'current_occupants':      room_info['current_occupants'],
                'roommate_details':       rm_details,
                'vacancies':              room_info['capacity'] - len(room_info['current_occupants']),
                'gender':                 room_info['gender'],
                'building':               room_info.get('building', ''),
                'floor':                  room_info.get('floor', ''),
                'resident_type':          room_info.get('resident_type', 'mixed'),
                'compatibility_score':    round(avg_sim, 3),
                'avg_roommate_similarity':round(avg_sim, 3),
                'reasons':                reasons,
                'recommendation_reasons': reasons,
            })

        room_scores.sort(key=lambda x: -x['compatibility_score'])
        return room_scores[:top_k]

    # ------------------------------------------------------------------
    def match_pair(self, id_a, id_b):
        pa = self.dl.get_student_profile(id_a)
        pb = self.dl.get_student_profile(id_b)
        if not pa or not pb:
            return None

        score   = self.ss.predict_similarity(pa, pb)
        details = self._match_details(pa, pb)

        reasons = []
        if details.get('university')    : reasons.append(f"same university ({details['university_a']})")
        if details.get('department')    : reasons.append(f"same department ({details['department_a']})")
        if details.get('ethnicity')     : reasons.append(f"same cultural background ({details['ethnicity_a']})")
        if details.get('home_city')     : reasons.append(f"from same city ({details['home_city_a']})")
        if details.get('personality')   : reasons.append(f"compatible personalities ({details['personality_a']})")
        if details.get('sleep_schedule'): reasons.append(f"same sleep schedule ({details['sleep_schedule_a']})")
        if details.get('food_preference'): reasons.append(f"same food preference ({details['food_preference_a']})")
        if details.get('study_similar') : reasons.append(f"similar study habits")
        if details.get('cleanliness_similar'): reasons.append(f"similar cleanliness")
        if not reasons:
            reasons.append("standard compatibility")

        return {
            'student_a':     id_a,
            'student_b':     id_b,
            'overall_score': score,
            'breakdown':     details,
            'reasons':       ", ".join(reasons),
        }

    # ------------------------------------------------------------------
    def _safe_float(self, val, default=0.5):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        try:
            return float(val)
        except:
            return default

    def _match_details(self, pa, pb):
        d = {}
        for key in ['university','department','degree','ethnicity','home_city',
                    'personality','sleep_schedule','food_preference','smoking_pref','guest_policy','gender']:
            va = pa.get(key) or ''
            vb = pb.get(key) or ''
            d[key]       = (va == vb and va not in ('', 'N/A'))
            d[key+'_a']  = va or 'N/A'
            d[key+'_b']  = vb or 'N/A'

        # Study habits
        sa = self._safe_float(pa.get('study_habits'), 0.5)
        sb = self._safe_float(pb.get('study_habits'), 0.5)
        d['study_similar']        = abs(sa - sb) < 0.2
        d['study_habits_a']       = f"{sa*100:.0f}%"
        d['study_habits_b']       = f"{sb*100:.0f}%"
        d['study_similarity_score']= round(1 - abs(sa - sb), 3)

        # Cleanliness
        def ck(p):
            v = p.get('cleanliness_level', 3)
            try: return int(float(v))
            except: return 3
        ca, cb = ck(pa), ck(pb)
        d['cleanliness_similar'] = abs(ca - cb) <= 1
        d['cleanliness_a']       = f"{ca}/5"
        d['cleanliness_b']       = f"{cb}/5"
        d['cleanliness_score']   = round(1 - abs(ca - cb) / 4, 3)

        # Social / noise
        for key in ['social_preference','noise_tolerance']:
            va = self._safe_float(pa.get(key), 0.5)
            vb = self._safe_float(pb.get(key), 0.5)
            d[key+'_similar'] = abs(va - vb) < 0.2
            d[key+'_a']       = f"{va*100:.0f}%"
            d[key+'_b']       = f"{vb*100:.0f}%"

        # Year
        ya = pa.get('year', 'N/A'); yb = pb.get('year', 'N/A')
        d['year_a'] = str(ya); d['year_b'] = str(yb)
        try:   d['year_diff'] = abs(int(ya) - int(yb))
        except: d['year_diff'] = 'N/A'

        return d

    def _room_reasons(self, my_profile, rm_details):
        reasons = []
        if not rm_details:
            return ["✨ Empty room – choose your own roommate!"]

        same_dept = sum(1 for r in rm_details if r.get('department') == my_profile.get('department'))
        if same_dept: reasons.append(f"📚 {same_dept} roommate(s) from same department")

        same_eth = sum(1 for r in rm_details if r.get('ethnicity') == my_profile.get('ethnicity'))
        if same_eth: reasons.append(f"🌍 {same_eth} roommate(s) from same cultural background")

        same_pers = sum(1 for r in rm_details if r.get('personality') == my_profile.get('personality'))
        if same_pers: reasons.append(f"💭 {same_pers} roommate(s) with similar personality")

        same_sleep = sum(1 for r in rm_details if r.get('sleep_schedule') == my_profile.get('sleep_schedule'))
        if same_sleep: reasons.append(f"😴 {same_sleep} roommate(s) with same sleep schedule")

        avg_sim = np.mean([r.get('similarity', 0) for r in rm_details])
        if avg_sim > 0.8:   reasons.append(f"⭐ Excellent compatibility ({avg_sim*100:.0f}%)")
        elif avg_sim > 0.6: reasons.append(f"👍 Good compatibility ({avg_sim*100:.0f}%)")

        if not reasons:
            reasons.append("🔄 Standard compatibility – potential for a good match")
        return reasons