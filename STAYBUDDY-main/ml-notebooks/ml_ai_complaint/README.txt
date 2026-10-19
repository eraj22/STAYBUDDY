HOW TO RUN
──────────
1. Open a NEW CMD window (separate from main AI server)

2. Navigate to this folder:
   cd STAYBUDDY-main\ml-notebooks\complaint_ai

3. Install requirements (once):
   pip install fastapi uvicorn scikit-learn joblib scipy

4. Run:
   python app_api.py

5. You will see:
   ✅ Complaint AI loaded — accuracy: 99.6%
   Running on: http://127.0.0.1:8001

6. Test at: http://127.0.0.1:8001/docs

NOTE: Main AI server runs on port 8000
      Complaint AI runs on port 8001
