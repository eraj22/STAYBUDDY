#!/usr/bin/env python3
# COMPLETE WORKING AI COMPLAINT SYSTEM - NO ERRORS
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from urllib.parse import urlparse
import re

# ============================================
# KNOWLEDGE BASE
# ============================================

COMPLAINT_CATEGORIES = {
    "Cleanliness": {
        "keywords": ["dirty", "clean", "cleaning", "dust", "bathroom", "toilet", "garbage", 
                    "trash", "sweep", "floor", "smell", "mosquito", "cockroach", "pest", "hygiene",
                    "filthy", "mess", "waste", "unclean", "stain"],
        "suggestions": [
            "Schedule immediate housekeeping for the reported area",
            "Arrange pest control inspection within 24 hours",
            "Add the area to daily cleaning checklist"
        ],
        "resolution_time": "1-2 days",
        "color": "#0A6B6E",
        "priority_boost": 0
    },
    "Maintenance": {
        "keywords": ["broken", "repair", "fix", "leak", "water", "pipe", "electricity", 
                    "fan", "light", "bulb", "switch", "door", "lock", "ac", "wifi", "not working",
                    "damaged", "fault", "problem", "outage"],
        "suggestions": [
            "Assign maintenance technician within 24 hours",
            "Log issue in maintenance register with priority tag",
            "Escalate to owner if not resolved within 48 hours"
        ],
        "resolution_time": "1-3 days",
        "color": "#B85C00",
        "priority_boost": 1
    },
    "Food Quality": {
        "keywords": ["food", "meal", "lunch", "dinner", "breakfast", "taste", "stale", "rotten", 
                    "cold", "hair", "quantity", "portion", "quality", "cook", "kitchen", "menu",
                    "undercooked", "raw", "spoiled"],
        "suggestions": [
            "Inspect kitchen and food storage immediately",
            "Check food quality with cook and supplier",
            "Conduct hygiene audit of kitchen area"
        ],
        "resolution_time": "Same day",
        "color": "#2D6A11",
        "priority_boost": 0
    },
    "Safety/Security": {
        "keywords": ["theft", "stolen", "missing", "security", "guard", "cctv", 
                    "unauthorized", "stranger", "unsafe", "dangerous", "emergency", "fire",
                    "harass", "threat", "intruder"],
        "suggestions": [
            "Investigate immediately and file incident report",
            "Review CCTV footage from reported time",
            "Increase security patrols in reported area"
        ],
        "resolution_time": "Immediate - 24 hours",
        "color": "#C0392B",
        "priority_boost": 2
    },
    "Noise/Disturbance": {
        "keywords": ["noise", "loud", "music", "shouting", "party", "sleep", 
                    "quiet", "disturb", "night", "late", "midnight", "screaming", "banging"],
        "suggestions": [
            "Warn the noisy party and issue formal notice",
            "Enforce quiet hours policy (10 PM - 7 AM)",
            "Mediate between affected students"
        ],
        "resolution_time": "Same day",
        "color": "#7B3FC4",
        "priority_boost": 0
    },
    "Staff Behavior": {
        "keywords": ["warden", "staff", "cook", "guard", "cleaner", "rude", "behavior",
                    "disrespect", "shout", "threat", "unfair", "biased", "abuse", 
                    "unprofessional", "misbehave", "attitude"],
        "suggestions": [
            "Counsel the staff member formally",
            "Document incident with date, time and witnesses",
            "Issue written warning to staff member"
        ],
        "resolution_time": "1-2 days",
        "color": "#185FA5",
        "priority_boost": 1
    },
    "Roommate Issue": {
        "keywords": ["roommate", "sharing", "personal", "space", "belongings", 
                    "borrow", "smoke", "fight", "conflict", "argument", "privacy", 
                    "guest", "visitor", "smoking"],
        "suggestions": [
            "Mediate between roommates with warden present",
            "Set clear room rules and document agreement",
            "Offer room change if mediation fails"
        ],
        "resolution_time": "2-3 days",
        "color": "#E91E8C",
        "priority_boost": 0
    },
    "Billing/Payment": {
        "keywords": ["bill", "billing", "charge", "money", "fee", "payment", "rent",
                    "extra", "refund", "deposit", "receipt", "invoice", "overcharge",
                    "deduction", "fine", "penalty"],
        "suggestions": [
            "Review fee structure and provide itemized receipt",
            "Cross-check payment records with management",
            "Process refund if charge was incorrect"
        ],
        "resolution_time": "3-5 days",
        "color": "#FFA500",
        "priority_boost": 0
    }
}

# ============================================
# CATEGORIZATION FUNCTION
# ============================================

def categorize_complaint(text):
    """Categorize complaint based on keywords"""
    if not text or not text.strip():
        return {
            "category": "Other",
            "subcategory": "General",
            "priority": "Low",
            "suggestion": "Please provide a valid complaint description",
            "confidence": 0,
            "suggested_actions": ["Review and respond within 48 hours"],
            "estimated_resolution_time": "2-3 days",
            "color": "#5F5E5A"
        }
    
    text_lower = text.lower()
    
    # Calculate scores for each category
    scores = {}
    matched_keywords = {}
    for category, data in COMPLAINT_CATEGORIES.items():
        score = 0
        matched = []
        for keyword in data["keywords"]:
            if keyword in text_lower:
                weight = 2 if " " in keyword else 1
                score += weight
                matched.append(keyword)
        if score > 0:
            scores[category] = score
            matched_keywords[category] = matched
    
    # Get best matching category
    if not scores:
        category = "Other"
        confidence = 0.5
        matched = []
        category_data = {"suggestions": ["Review and respond within 48 hours"], 
                        "resolution_time": "2-3 days", "color": "#5F5E5A", "priority_boost": 0}
    else:
        category = max(scores, key=scores.get)
        matched = matched_keywords.get(category, [])
        confidence = min(0.95, scores[category] / sum(scores.values()))
        category_data = COMPLAINT_CATEGORIES[category]
    
    # Determine priority
    high_priority = ["urgent", "immediately", "emergency", "stolen", "missing", "fire", 
                     "theft", "broken", "leak", "dangerous", "injury", "blocked"]
    low_priority = ["small", "minor", "little", "slightly", "prefer", "suggest", "recommend"]
    
    if any(word in text_lower for word in high_priority) or category_data.get("priority_boost", 0) >= 2:
        priority = "High"
    elif any(word in text_lower for word in low_priority) or category_data.get("priority_boost", 0) == 0:
        priority = "Low"
    else:
        priority = "Medium"
    
    # Find subcategory
    subcategory = "General"
    sub_map = {
        "pest": "Pest Control", "mosquito": "Pest Control", "cockroach": "Pest Control",
        "bathroom": "Bathroom", "toilet": "Bathroom", "washroom": "Bathroom",
        "electrical": "Electrical", "fan": "Electrical", "light": "Electrical",
        "plumbing": "Plumbing", "pipe": "Plumbing", "leak": "Plumbing",
        "taste": "Taste/Quality", "stale": "Taste/Quality", "rotten": "Taste/Quality"
    }
    for key, val in sub_map.items():
        if key in text_lower:
            subcategory = val
            break
    
    # Get suggestion
    suggestion = category_data["suggestions"][0]
    for s in category_data["suggestions"]:
        if any(kw in s.lower() for kw in matched[:2]):
            suggestion = s
            break
    
    if priority == "High":
        suggestion = f"URGENT: {suggestion}"
    
    return {
        "category": category,
        "subcategory": subcategory,
        "priority": priority,
        "suggestion": suggestion,
        "confidence": round(confidence, 2),
        "suggested_actions": category_data["suggestions"],
        "estimated_resolution_time": category_data["resolution_time"],
        "color": category_data["color"],
        "matched_keywords": matched[:5]
    }

# ============================================
# HTTP SERVER HANDLER
# ============================================

class ComplaintHandler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        if parsed.path == '/' or parsed.path == '/dashboard':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_dashboard().encode())
        
        elif parsed.path == '/api/categories':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            response = {"success": True, "categories": list(COMPLAINT_CATEGORIES.keys())}
            self.wfile.write(json.dumps(response).encode())
        
        elif parsed.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {"status": "healthy", "timestamp": datetime.now().isoformat()}
            self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        if self.path == '/api/analyze':
            content_len = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_len)
            
            try:
                data = json.loads(post_data.decode())
                complaint_text = data.get('complaint_text', '')
                
                if not complaint_text:
                    response = {"success": False, "error": "No complaint text provided"}
                else:
                    result = categorize_complaint(complaint_text)
                    result["success"] = True
                    result["complaint_id"] = f"CMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    response = result
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Invalid JSON"}).encode())
        
        elif self.path == '/api/bulk':
            content_len = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_len)
            
            try:
                data = json.loads(post_data.decode())
                complaints = data.get('complaints', [])
                results = []
                for c in complaints:
                    text = c.get('complaint_text', '')
                    if text:
                        results.append(categorize_complaint(text))
                
                response = {"success": True, "total": len(results), "results": results}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_dashboard(self):
        return '''<!DOCTYPE html>
<html>
<head>
    <title>AI Complaint System</title>
    <meta charset="UTF-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { background: white; border-radius: 15px; padding: 25px; margin-bottom: 25px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        h1 { color: #333; margin-bottom: 10px; }
        .subtitle { color: #666; }
        .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; }
        .card { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .card h2 { margin-bottom: 20px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
        textarea { width: 100%; padding: 15px; border: 2px solid #ddd; border-radius: 10px; font-size: 14px; font-family: monospace; resize: vertical; margin-bottom: 15px; }
        button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; transition: transform 0.2s; }
        button:hover { transform: translateY(-2px); }
        .result { margin-top: 20px; padding: 15px; border-radius: 10px; background: #f8f9fa; display: none; }
        .category-badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; margin: 5px; }
        .priority-High { background: #fee; color: #c00; padding: 3px 10px; border-radius: 15px; display: inline-block; }
        .priority-Medium { background: #fef9e6; color: #e67e22; padding: 3px 10px; border-radius: 15px; display: inline-block; }
        .priority-Low { background: #e8f5e9; color: #27ae60; padding: 3px 10px; border-radius: 15px; display: inline-block; }
        .suggestion-box { background: #e8f4f8; padding: 15px; border-radius: 10px; margin-top: 15px; border-left: 4px solid #667eea; }
        .loading { display: none; text-align: center; padding: 20px; }
        .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .example-buttons { margin-top: 15px; }
        .example-btn { background: #f0f0f0; color: #333; padding: 5px 10px; margin: 5px; font-size: 12px; border: none; cursor: pointer; border-radius: 5px; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; margin-bottom: 20px; }
        .stat { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 10px; }
        .stat-value { font-size: 24px; font-weight: bold; color: #667eea; }
        .stat-label { font-size: 12px; color: #666; margin-top: 5px; }
        @media (max-width: 768px) { .main-content { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AI Complaint Management System</h1>
            <p class="subtitle">Automated categorization & intelligent suggestions</p>
        </div>
        <div class="main-content">
            <div class="card">
                <h2>📝 Analyze Complaint</h2>
                <textarea id="complaintText" rows="5" placeholder="Enter complaint description here..."></textarea>
                <button onclick="analyzeComplaint()">🔍 Analyze & Get Suggestions</button>
                <div class="example-buttons">
                    <button class="example-btn" onclick="setExample('The bathroom is very dirty and has cockroaches')">🧹 Cleanliness</button>
                    <button class="example-btn" onclick="setExample('My fan is broken and not working')">🔧 Maintenance</button>
                    <button class="example-btn" onclick="setExample('The food is stale and tastes bad')">🍝 Food Quality</button>
                    <button class="example-btn" onclick="setExample('Someone stole my phone from the room')">🔒 Safety</button>
                </div>
                <div id="result" class="result"></div>
                <div id="loading" class="loading"><div class="spinner"></div><p>Analyzing...</p></div>
            </div>
            <div class="card">
                <h2>ℹ️ System Features</h2>
                <div class="stats">
                    <div class="stat"><div class="stat-value">8+</div><div class="stat-label">Categories</div></div>
                    <div class="stat"><div class="stat-value">AI</div><div class="stat-label">Powered</div></div>
                    <div class="stat"><div class="stat-value">24/7</div><div class="stat-label">Available</div></div>
                </div>
                <h3>🎯 Categories:</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>✓ Cleanliness</li><li>✓ Maintenance</li><li>✓ Food Quality</li>
                    <li>✓ Safety/Security</li><li>✓ Noise/Disturbance</li>
                    <li>✓ Staff Behavior</li><li>✓ Roommate Issue</li><li>✓ Billing/Payment</li>
                </ul>
                <h3 style="margin-top: 20px;">✨ Features:</h3>
                <ul style="margin-left: 20px; margin-top: 10px;">
                    <li>✓ AI-powered categorization</li>
                    <li>✓ Priority assignment (High/Medium/Low)</li>
                    <li>✓ Smart suggestions & actions</li>
                    <li>✓ Resolution time estimates</li>
                    <li>✓ Confidence scoring</li>
                </ul>
            </div>
        </div>
    </div>
    <script>
        async function analyzeComplaint() {
            const text = document.getElementById('complaintText').value;
            if (!text.trim()) { alert('Please enter a complaint'); return; }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ complaint_text: text })
                });
                const data = await response.json();
                
                if (data.success) {
                    const priorityClass = `priority-${data.priority}`;
                    document.getElementById('result').innerHTML = `
                        <h3>✅ Analysis Result</h3>
                        <div style="margin: 10px 0;">
                            <span class="category-badge" style="background: ${data.color}20; color: ${data.color}">
                                📁 ${data.category}
                            </span>
                            <span class="category-badge" style="background: #f0f0f0;">
                                🔍 ${data.subcategory}
                            </span>
                            <span class="${priorityClass}">⚡ ${data.priority} Priority</span>
                        </div>
                        <div class="suggestion-box">
                            <strong>💡 AI Suggestion:</strong><br>
                            ${data.suggestion}
                        </div>
                        <div style="margin-top: 15px;">
                            <strong>✅ Suggested Actions:</strong>
                            <ul>${data.suggested_actions.map(a => `<li>${a}</li>`).join('')}</ul>
                        </div>
                        <div style="margin-top: 10px; padding: 10px; background: #e8f5e9; border-radius: 8px;">
                            ⏱️ <strong>Estimated resolution time:</strong> ${data.estimated_resolution_time}
                        </div>
                        <div style="margin-top: 10px; font-size: 12px; color: #999;">
                            Confidence: ${Math.round(data.confidence * 100)}% | Complaint ID: ${data.complaint_id}
                        </div>
                    `;
                    document.getElementById('result').style.display = 'block';
                } else {
                    document.getElementById('result').innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                    document.getElementById('result').style.display = 'block';
                }
            } catch(e) {
                document.getElementById('result').innerHTML = `<div style="color: red;">Error: ${e.message}</div>`;
                document.getElementById('result').style.display = 'block';
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        function setExample(text) {
            document.getElementById('complaintText').value = text;
            analyzeComplaint();
        }
    </script>
</body>
</html>'''
    
    def log_message(self, format, *args):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {args[0]}")

# ============================================
# MAIN
# ============================================

def run_server(port=8003):
    server = HTTPServer(('', port), ComplaintHandler)
    print("=" * 60)
    print("🤖 AI COMPLAINT CATEGORIZATION SYSTEM")
    print("=" * 60)
    print(f"✅ Server running at: http://127.0.0.1:{port}")
    print(f"📊 Dashboard: http://127.0.0.1:{port}/dashboard")
    print(f"🔧 API Endpoint: http://127.0.0.1:{port}/api/analyze")
    print("=" * 60)
    print("📋 Features:")
    print("   • 8+ Categories (Cleanliness, Maintenance, Food, Safety, etc.)")
    print("   • Priority Assignment (High/Medium/Low)")
    print("   • AI-Powered Suggestions")
    print("   • Resolution Time Estimates")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    run_server()