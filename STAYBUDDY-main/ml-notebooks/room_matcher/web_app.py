# web_app.py - FIXED VERSION
"""
Web-based AI Room Matching System
Supports BOTH students (STU-) and working professionals (JOB-).
Run:  python web_app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader    import DataLoader
from feature_engineering import FeatureEngineer
from similarity_scorer   import SimilarityScorer
from room_matcher   import RoomMatcher
from recommender    import RoomRecommender
from config         import AVAILABLE_ROOMS

app = Flask(__name__)
app.secret_key = 'room_matcher_secret_2024'

# ── CORS: allow Flutter web (any localhost port) to call this API ────────────
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    return response

@app.before_request
def handle_options():
    if request.method == 'OPTIONS':
        from flask import Response
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return res, 200

# ── Initialise ──────────────────────────────────────────────────────────────
print("🔧 Initialising AI Room Matching System…")
data_loader = DataLoader()
if not data_loader.load_data():
    print("❌ Failed to load data. Check CSV files.")
    sys.exit(1)

feature_engineer  = FeatureEngineer()
similarity_scorer = SimilarityScorer(feature_engineer)

if not similarity_scorer.load_model():
    print("📊 Training similarity model…")
    # Train on ALL profiles (students + professionals)
    X, y = similarity_scorer.prepare_training_data(
        data_loader.matches_df,
        data_loader.all_profiles       # ← use combined pool
    )
    if len(X) > 0:
        similarity_scorer.train(X, y)
        similarity_scorer.save_model()
    else:
        print("⚠️  Using rule-based matching (no training pairs found)")

room_matcher = RoomMatcher(data_loader, feature_engineer, similarity_scorer)
recommender  = RoomRecommender(room_matcher)
print("✅ System ready!")


# ── Helpers ──────────────────────────────────────────────────────────────────
def _clean(val, default='N/A'):
    """Return a JSON-safe scalar."""
    if val is None:
        return default
    if isinstance(val, float) and (np.isnan(val) or not np.isfinite(val)):
        return default
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 4)
    if isinstance(val, bool):
        return val
    return val

def _clean_profile(profile: dict) -> dict:
    return {k: _clean(v) for k, v in profile.items()}

def _clean_list(lst):
    out = []
    for item in lst:
        if isinstance(item, dict):
            cleaned = {}
            for k, v in item.items():
                if isinstance(v, dict):
                    cleaned[k] = {sk: _clean(sv) for sk, sv in v.items()}
                elif isinstance(v, list):
                    cleaned[k] = [_clean(x) for x in v]
                else:
                    cleaned[k] = _clean(v)
            out.append(cleaned)
        else:
            out.append(_clean(item))
    return out


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/student/<resident_id>')
def student_profile(resident_id):
    profile = data_loader.get_student_profile(resident_id)
    if profile:
        return jsonify(_clean_profile(profile))
    return jsonify({'error': 'Resident not found'}), 404


@app.route('/api/all_students')
def all_students():
    """Return all residents (students + professionals) for the dropdown."""
    try:
        resident_list = []
        for rid in data_loader.get_all_residents():
            p = data_loader.get_student_profile(rid)
            if not p:
                continue
            is_pro = bool(p.get('is_professional')) or str(rid).startswith('JOB-')
            dept   = _clean(p.get('department'), 'N/A')
            resident_list.append({
                'id':             rid,
                'gender':         _clean(p.get('gender'), 'N/A'),
                'department':     dept,
                'year':           _clean(p.get('year'), 'N/A'),
                'university':     str(_clean(p.get('university'), 'N/A'))[:40],
                'ethnicity':      _clean(p.get('ethnicity'), 'N/A'),
                'home_city':      _clean(p.get('home_city'), 'N/A'),
                'personality':    _clean(p.get('personality'), 'N/A'),
                'is_professional':is_pro,
                'job_title':      _clean(p.get('job_title'), 'Student') if is_pro else 'Student',
            })
        stu = sum(1 for r in resident_list if not r['is_professional'])
        pro = sum(1 for r in resident_list if r['is_professional'])
        print(f"✅ /api/all_students → {len(resident_list)} (STU:{stu} JOB:{pro})")
        return jsonify(resident_list)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stats')
def stats():
    try:
        all_ids   = data_loader.get_all_residents()
        stu_count = len(data_loader.get_students_only())
        pro_count = len(data_loader.get_professionals_only())
        unis, depts, eths = set(), set(), set()
        for rid in all_ids:
            p = data_loader.get_student_profile(rid)
            if not p: continue
            u = p.get('university','')
            if u and u not in ('Working Professional','N/A'): unis.add(u)
            d = p.get('department','')
            if d and d != 'N/A': depts.add(d)
            e = p.get('ethnicity','')
            if e and e != 'N/A': eths.add(e)
        return jsonify({
            'total_students':     len(all_ids),
            'student_count':      stu_count,
            'professional_count': pro_count,
            'total_universities': len(unis),
            'total_departments':  len(depts),
            'total_ethnicities':  len(eths),
            'available_rooms':    len(AVAILABLE_ROOMS),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_roommates', methods=['POST'])
def find_roommates():
    try:
        data       = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        top_k      = int(data.get('top_k', 5))
        pool       = data.get('search_pool', 'all')   # all | students | professionals

        if not resident_id:
            return jsonify({'error': 'resident_id required'}), 400

        profile = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': f'Resident {resident_id} not found'}), 404

        roommates = room_matcher.find_best_roommates(resident_id, top_k=top_k, search_pool=pool)

        return jsonify({
            'success':      True,
            'resident_id':  resident_id,
            'is_professional': bool(profile.get('is_professional')),
            'student_name': f"{profile.get('gender','')} – {profile.get('department','')}",
            'roommates':    _clean_list(roommates),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_rooms', methods=['POST'])
def find_rooms():
    try:
        data        = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        if not resident_id:
            return jsonify({'error': 'resident_id required'}), 400

        profile = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': f'Resident {resident_id} not found'}), 404

        rooms = room_matcher.find_best_rooms(resident_id, AVAILABLE_ROOMS, top_k=10)
        return jsonify({'success': True, 'resident_id': resident_id, 'rooms': _clean_list(rooms)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/check_compatibility', methods=['POST'])
def check_compatibility():
    try:
        data = request.get_json() or {}
        a    = data.get('student_a') or data.get('resident_a')
        b    = data.get('student_b') or data.get('resident_b')
        if not a or not b:
            return jsonify({'error': 'Two resident IDs required'}), 400

        match = room_matcher.match_pair(a, b)
        if match:
            match['overall_score'] = round(match['overall_score'], 3)
            if 'breakdown' in match:
                match['breakdown'] = {k: _clean(v) for k, v in match['breakdown'].items()}
            return jsonify(match)
        return jsonify({'error': 'One or both residents not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/filter_rooms', methods=['POST'])
def filter_rooms():
    try:
        data        = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        preferences = data.get('preferences', {})

        if not resident_id:
            return jsonify({'error': 'resident_id required'}), 400

        profile = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': f'Resident {resident_id} not found'}), 404

        # Filter rooms
        filtered = {}
        for room_id, info in AVAILABLE_ROOMS.items():
            vacancies = info['capacity'] - len(info['current_occupants'])
            if vacancies <= 0:
                continue

            if preferences.get('gender'):
                if info['gender'].lower() not in (preferences['gender'].lower(), 'mixed'):
                    continue

            if preferences.get('room_type'):
                if info['type'].lower() != preferences['room_type'].lower():
                    continue

            if preferences.get('resident_type'):
                rt = preferences['resident_type'].lower()
                if info.get('resident_type', 'mixed').lower() not in (rt, 'mixed'):
                    continue

            if preferences.get('university'):
                uni_filter = preferences['university'].lower()
                has_uni = any(
                    (data_loader.get_student_profile(occ) or {}).get('university','').lower() == uni_filter
                    for occ in info['current_occupants']
                )
                if len(info['current_occupants']) > 0 and not has_uni:
                    continue

            filtered[room_id] = info

        if not filtered:
            return jsonify({'success': True, 'resident_id': resident_id,
                            'rooms': [], 'message': 'No rooms match your filters.'})

        rooms = room_matcher.find_best_rooms(resident_id, filtered, top_k=10)
        return jsonify({'success': True, 'resident_id': resident_id,
                        'preferences': preferences, 'rooms': _clean_list(rooms)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_by_culture', methods=['POST'])
def find_by_culture():
    try:
        data        = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        profile     = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': 'Resident not found'}), 404

        ethnicity = profile.get('ethnicity','')
        if not ethnicity or ethnicity == 'N/A':
            return jsonify({'error': 'Ethnicity not specified'}), 400

        matches = []
        for oid, op in data_loader.all_profiles.items():
            if oid == resident_id or op.get('ethnicity') != ethnicity:
                continue
            sim = similarity_scorer.predict_similarity(profile, op)
            matches.append({
                'student_id':      oid,
                'similarity_score':round(sim, 3),
                'department':      _clean(op.get('department'),'N/A'),
                'home_city':       _clean(op.get('home_city'),'N/A'),
                'personality':     _clean(op.get('personality'),'N/A'),
                'ethnicity':       ethnicity,
                'sleep_schedule':  _clean(op.get('sleep_schedule'),'N/A'),
                'university':      _clean(op.get('university'),'N/A'),
                'is_professional': bool(op.get('is_professional')),
                'job_title':       _clean(op.get('job_title'),'Student'),
            })
        matches.sort(key=lambda x: -x['similarity_score'])
        return jsonify({'success': True, 'ethnicity': ethnicity, 'matches': matches[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_by_department', methods=['POST'])
def find_by_department():
    try:
        data        = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        profile     = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': 'Resident not found'}), 404

        dept = profile.get('department','')
        if not dept:
            return jsonify({'error': 'Department not specified'}), 400

        matches = []
        for oid, op in data_loader.all_profiles.items():
            if oid == resident_id or op.get('department') != dept:
                continue
            sim = similarity_scorer.predict_similarity(profile, op)
            matches.append({
                'student_id':      oid,
                'similarity_score':round(sim, 3),
                'department':      dept,
                'ethnicity':       _clean(op.get('ethnicity'),'N/A'),
                'home_city':       _clean(op.get('home_city'),'N/A'),
                'personality':     _clean(op.get('personality'),'N/A'),
                'sleep_schedule':  _clean(op.get('sleep_schedule'),'N/A'),
                'university':      _clean(op.get('university'),'N/A'),
                'is_professional': bool(op.get('is_professional')),
                'job_title':       _clean(op.get('job_title'),'Student'),
            })
        matches.sort(key=lambda x: -x['similarity_score'])
        return jsonify({'success': True, 'department': dept, 'matches': matches[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_by_hometown', methods=['POST'])
def find_by_hometown():
    try:
        data        = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        profile     = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': 'Resident not found'}), 404

        city = profile.get('home_city','')
        if not city or city == 'N/A':
            return jsonify({'error': 'Home city not specified'}), 400

        matches = []
        for oid, op in data_loader.all_profiles.items():
            if oid == resident_id or op.get('home_city') != city:
                continue
            sim = similarity_scorer.predict_similarity(profile, op)
            matches.append({
                'student_id':      oid,
                'similarity_score':round(sim, 3),
                'department':      _clean(op.get('department'),'N/A'),
                'ethnicity':       _clean(op.get('ethnicity'),'N/A'),
                'home_city':       city,
                'personality':     _clean(op.get('personality'),'N/A'),
                'sleep_schedule':  _clean(op.get('sleep_schedule'),'N/A'),
                'university':      _clean(op.get('university'),'N/A'),
                'is_professional': bool(op.get('is_professional')),
            })
        matches.sort(key=lambda x: -x['similarity_score'])
        return jsonify({'success': True, 'hometown': city, 'matches': matches[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_by_personality', methods=['POST'])
def find_by_personality():
    try:
        data        = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        profile     = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': 'Resident not found'}), 404

        pers = profile.get('personality','')
        if not pers:
            return jsonify({'error': 'Personality not specified'}), 400

        matches = []
        for oid, op in data_loader.all_profiles.items():
            if oid == resident_id or op.get('personality') != pers:
                continue
            sim = similarity_scorer.predict_similarity(profile, op)
            matches.append({
                'student_id':      oid,
                'similarity_score':round(sim, 3),
                'department':      _clean(op.get('department'),'N/A'),
                'ethnicity':       _clean(op.get('ethnicity'),'N/A'),
                'home_city':       _clean(op.get('home_city'),'N/A'),
                'sleep_schedule':  _clean(op.get('sleep_schedule'),'N/A'),
                'is_professional': bool(op.get('is_professional')),
            })
        matches.sort(key=lambda x: -x['similarity_score'])
        return jsonify({'success': True, 'personality': pers, 'matches': matches[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_professionals', methods=['GET'])
def find_professionals():
    try:
        pros = []
        for pid, p in data_loader.professional_profiles.items():
            pros.append({
                'student_id':   pid,
                'job_title':    _clean(p.get('job_title'),'Professional'),
                'department':   _clean(p.get('department'),'N/A'),
                'gender':       _clean(p.get('gender'),'N/A'),
                'ethnicity':    _clean(p.get('ethnicity'),'N/A'),
                'home_city':    _clean(p.get('home_city'),'N/A'),
                'work_schedule':_clean(p.get('work_schedule'),'Regular hours'),
            })
        return jsonify({'success': True, 'total_professionals': len(pros), 'professionals': pros})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/find_compatible_professionals', methods=['POST'])
def find_compatible_professionals():
    try:
        data        = request.get_json() or {}
        resident_id = data.get('student_id') or data.get('resident_id')
        profile     = data_loader.get_student_profile(resident_id)
        if not profile:
            return jsonify({'error': 'Resident not found'}), 404

        results = []
        for pid, pp in data_loader.professional_profiles.items():
            if pid == resident_id:
                continue
            sim = similarity_scorer.predict_similarity(profile, pp)
            if sim >= 0.5:
                results.append({
                    'student_id':      pid,
                    'similarity_score':round(sim, 3),
                    'job_title':       _clean(pp.get('job_title'),'Professional'),
                    'department':      _clean(pp.get('department'),'N/A'),
                    'ethnicity':       _clean(pp.get('ethnicity'),'N/A'),
                    'personality':     _clean(pp.get('personality'),'N/A'),
                    'work_schedule':   _clean(pp.get('work_schedule'),'Regular hours'),
                })
        results.sort(key=lambda x: -x['similarity_score'])
        return jsonify({'success': True, 'resident_id': resident_id, 'matches': results[:10]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/residents_by_type')
def residents_by_type():
    rtype = request.args.get('type', 'all')
    if rtype == 'students':
        ids = data_loader.get_students_only()
    elif rtype == 'professionals':
        ids = data_loader.get_professionals_only()
    else:
        ids = data_loader.get_all_residents()

    out = []
    for rid in ids[:200]:
        p = data_loader.get_student_profile(rid)
        if p:
            out.append({
                'id':         rid,
                'type':       'professional' if p.get('is_professional') else 'student',
                'job_title':  _clean(p.get('job_title'),'Student'),
                'department': _clean(p.get('department'),'N/A'),
                'gender':     _clean(p.get('gender'),'N/A'),
            })
    return jsonify({'success': True, 'count': len(out), 'residents': out})


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏠 AI ROOM MATCHING SYSTEM – WEB INTERFACE")
    print("="*60)
    all_r = data_loader.get_all_residents()
    print(f"📊 {len(all_r)} residents loaded  "
          f"(STU: {len(data_loader.get_students_only())}  "
          f"JOB: {len(data_loader.get_professionals_only())})")
    print("📱 http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, port=5000, host='127.0.0.1')