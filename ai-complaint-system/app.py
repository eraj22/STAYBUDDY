# ai-complaint-system/app.py
# ══════════════════════════════════════════════════════════════════════════════
# AI COMPLAINT CATEGORIZATION & SUGGESTION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import pandas as pd
import json
import os
from pathlib import Path
import uvicorn

# Import the complaint analyzer
from complaint_analyzer import ComplaintAnalyzer

# Create FastAPI app
app = FastAPI(
    title="AI Complaint Categorization & Suggestion System",
    description="Automatically categorizes complaints and provides AI-powered suggestions",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the analyzer
analyzer = ComplaintAnalyzer()

# Define data models
class ComplaintInput(BaseModel):
    complaint_text: str
    student_id: Optional[str] = None
    hostel_id: Optional[str] = None

class BulkComplaintInput(BaseModel):
    complaints: List[ComplaintInput]

class ComplaintResponse(BaseModel):
    success: bool
    complaint_id: Optional[str] = None
    category: Optional[str] = None
    subcategory: Optional[str] = None
    priority: Optional[str] = None
    ai_suggestion: Optional[str] = None
    confidence: Optional[float] = None
    similar_complaints: Optional[List[Dict]] = None
    suggested_actions: Optional[List[str]] = None
    estimated_resolution_time: Optional[str] = None
    error: Optional[str] = None

# ══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Complaint Categorization & Suggestion System",
        "version": "2.0.0",
        "status": "active",
        "endpoints": {
            "POST /analyze": "Analyze a single complaint",
            "POST /bulk-analyze": "Analyze multiple complaints",
            "POST /upload-csv": "Upload and analyze CSV file",
            "GET /statistics": "Get complaint statistics",
            "GET /trends": "Get complaint trends over time",
            "GET /categories": "Get all categories",
            "POST /suggestions": "Get AI suggestions for a category",
            "GET /dashboard": "View web dashboard"
        }
    }

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard():
    """Web dashboard for complaint management"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Complaint Management Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
            .container { max-width: 1400px; margin: 0 auto; }
            .header { background: white; border-radius: 15px; padding: 25px; margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            h1 { color: #333; margin-bottom: 10px; }
            .subtitle { color: #666; margin-bottom: 20px; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 25px; }
            .stat-card { background: white; border-radius: 10px; padding: 20px; text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .stat-number { font-size: 2em; font-weight: bold; color: #667eea; }
            .stat-label { color: #666; margin-top: 10px; }
            .main-content { display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 25px; }
            .card { background: white; border-radius: 15px; padding: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            .card h2 { margin-bottom: 20px; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }
            textarea { width: 100%; padding: 15px; border: 2px solid #ddd; border-radius: 10px; font-size: 14px; font-family: monospace; resize: vertical; margin-bottom: 15px; }
            button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 12px 30px; border-radius: 8px; cursor: pointer; font-size: 16px; transition: transform 0.2s; }
            button:hover { transform: translateY(-2px); }
            .result { margin-top: 20px; padding: 15px; border-radius: 10px; background: #f8f9fa; display: none; }
            .category-badge { display: inline-block; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; margin: 5px; }
            .priority { font-weight: bold; padding: 3px 10px; border-radius: 15px; display: inline-block; }
            .priority-High { background: #fee; color: #c00; }
            .priority-Medium { background: #fef9e6; color: #e67e22; }
            .priority-Low { background: #e8f5e9; color: #27ae60; }
            .suggestion-box { background: #e8f4f8; padding: 15px; border-radius: 10px; margin-top: 15px; border-left: 4px solid #667eea; }
            .complaint-list { max-height: 400px; overflow-y: auto; }
            .complaint-item { border-bottom: 1px solid #eee; padding: 10px; cursor: pointer; transition: background 0.2s; }
            .complaint-item:hover { background: #f8f9fa; }
            .loading { display: none; text-align: center; padding: 20px; }
            .spinner { border: 3px solid #f3f3f3; border-top: 3px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
            @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
            @media (max-width: 768px) { .main-content { grid-template-columns: 1fr; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 AI Complaint Management System</h1>
                <p class="subtitle">Automated categorization & intelligent suggestions for hostel complaints</p>
            </div>
            
            <div class="stats-grid" id="statsGrid">
                <div class="stat-card"><div class="stat-number" id="totalComplaints">-</div><div class="stat-label">Total Complaints</div></div>
                <div class="stat-card"><div class="stat-number" id="avgResponse">-</div><div class="stat-label">Avg Response Time</div></div>
                <div class="stat-card"><div class="stat-number" id="resolution">-</div><div class="stat-label">Resolution Rate</div></div>
                <div class="stat-card"><div class="stat-number" id="categories">-</div><div class="stat-label">Categories</div></div>
            </div>
            
            <div class="main-content">
                <div class="card">
                    <h2>📝 Analyze New Complaint</h2>
                    <textarea id="complaintText" rows="5" placeholder="Enter complaint description here..."></textarea>
                    <button onclick="analyzeComplaint()">🔍 Categorize & Get Suggestions</button>
                    <div id="analysisResult" class="result"></div>
                    <div id="loading" class="loading"><div class="spinner"></div><p>Analyzing...</p></div>
                </div>
                
                <div class="card">
                    <h2>📊 Recent Complaints</h2>
                    <div class="complaint-list" id="recentComplaints"></div>
                </div>
            </div>
            
            <div class="card" id="trendsCard" style="display:none;">
                <h2>📈 Category Distribution</h2>
                <canvas id="categoryChart" style="max-height: 300px;"></canvas>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
            let categoryChart = null;
            
            async function loadStatistics() {
                try {
                    const response = await fetch('/statistics');
                    const data = await response.json();
                    if (data.success) {
                        document.getElementById('totalComplaints').textContent = data.total_complaints || 0;
                        document.getElementById('categories').textContent = Object.keys(data.category_distribution || {}).length;
                        
                        if (categoryChart && data.category_distribution) {
                            categoryChart.data.datasets[0].data = Object.values(data.category_distribution);
                            categoryChart.update();
                            document.getElementById('trendsCard').style.display = 'block';
                        }
                    }
                } catch(e) { console.error(e); }
            }
            
            async function loadRecentComplaints() {
                try {
                    const response = await fetch('/trends');
                    const data = await response.json();
                    if (data.success && data.recent_complaints) {
                        const container = document.getElementById('recentComplaints');
                        container.innerHTML = data.recent_complaints.map(c => `
                            <div class="complaint-item" onclick="fillComplaint('${c.complaint_text.replace(/'/g, "\\'")}')">
                                <strong>${c.category || 'Unknown'}</strong> - ${c.priority || 'Medium'} Priority<br>
                                <small>${c.complaint_text.substring(0, 100)}...</small>
                            </div>
                        `).join('');
                    }
                } catch(e) { console.error(e); }
            }
            
            async function analyzeComplaint() {
                const text = document.getElementById('complaintText').value;
                if (!text.trim()) {
                    alert('Please enter a complaint');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('analysisResult').style.display = 'none';
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ complaint_text: text })
                    });
                    const data = await response.json();
                    
                    if (data.success) {
                        const priorityClass = `priority-${data.priority}`;
                        document.getElementById('analysisResult').innerHTML = `
                            <h3>Analysis Result</h3>
                            <div>
                                <span class="category-badge" style="background: ${data.color || '#667eea'}20; color: ${data.color || '#667eea'}">
                                    📁 ${data.category}
                                </span>
                                <span class="category-badge" style="background: #f0f0f0;">
                                    🔍 ${data.subcategory}
                                </span>
                                <span class="priority ${priorityClass}">
                                    ⚡ ${data.priority} Priority
                                </span>
                            </div>
                            <div class="suggestion-box">
                                <strong>💡 AI Suggestion:</strong><br>
                                ${data.ai_suggestion || data.suggestion || 'Review and respond within 48 hours'}
                            </div>
                            ${data.suggested_actions ? `
                            <div style="margin-top: 15px;">
                                <strong>✅ Suggested Actions:</strong>
                                <ul>${data.suggested_actions.map(a => `<li>${a}</li>`).join('')}</ul>
                            </div>
                            ` : ''}
                            ${data.estimated_resolution_time ? `
                            <div style="margin-top: 10px; color: #666;">
                                ⏱️ Estimated resolution: ${data.estimated_resolution_time}
                            </div>
                            ` : ''}
                            <div style="margin-top: 10px; font-size: 12px; color: #999;">
                                Confidence: ${Math.round((data.confidence || 0.8) * 100)}%
                            </div>
                        `;
                        document.getElementById('analysisResult').style.display = 'block';
                    } else {
                        document.getElementById('analysisResult').innerHTML = `<div style="color: red;">Error: ${data.error}</div>`;
                        document.getElementById('analysisResult').style.display = 'block';
                    }
                } catch(e) {
                    document.getElementById('analysisResult').innerHTML = `<div style="color: red;">Error: ${e.message}</div>`;
                    document.getElementById('analysisResult').style.display = 'block';
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            function fillComplaint(text) {
                document.getElementById('complaintText').value = text;
                analyzeComplaint();
            }
            
            // Initialize
            loadStatistics();
            loadRecentComplaints();
            
            // Initialize chart
            const ctx = document.getElementById('categoryChart')?.getContext('2d');
            if (ctx) {
                categoryChart = new Chart(ctx, {
                    type: 'bar',
                    data: { labels: [], datasets: [{ label: 'Number of Complaints', data: [], backgroundColor: '#667eea' }] },
                    options: { responsive: true, scales: { y: { beginAtZero: true } } }
                });
            }
            
            // Auto-refresh every 30 seconds
            setInterval(() => { loadStatistics(); loadRecentComplaints(); }, 30000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/analyze", response_model=ComplaintResponse)
async def analyze_complaint(complaint: ComplaintInput):
    """Analyze a single complaint and get AI suggestions"""
    try:
        if not complaint.complaint_text or not complaint.complaint_text.strip():
            return ComplaintResponse(success=False, error="Complaint text cannot be empty")
        
        result = analyzer.analyze_complaint(complaint.complaint_text)
        
        return ComplaintResponse(
            success=True,
            complaint_id=result.get("complaint_id"),
            category=result.get("category"),
            subcategory=result.get("subcategory"),
            priority=result.get("priority"),
            ai_suggestion=result.get("suggestion"),
            confidence=result.get("confidence", 0.85),
            similar_complaints=result.get("similar_complaints", []),
            suggested_actions=result.get("suggested_actions", []),
            estimated_resolution_time=result.get("estimated_resolution_time")
        )
    except Exception as e:
        return ComplaintResponse(success=False, error=str(e))

@app.post("/bulk-analyze")
async def bulk_analyze(complaints: BulkComplaintInput):
    """Analyze multiple complaints at once"""
    try:
        results = []
        for complaint in complaints.complaints:
            result = analyzer.analyze_complaint(complaint.complaint_text)
            results.append(result)
        
        # Generate summary
        category_summary = Counter(r.get("category", "Other") for r in results)
        
        return {
            "success": True,
            "total": len(results),
            "results": results,
            "summary": dict(category_summary)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and analyze complaints from CSV file"""
    try:
        # Read CSV
        df = pd.read_csv(file.file)
        
        # Find complaint text column
        text_col = None
        for col in ['complaint_text', 'text', 'complaint', 'description']:
            if col in df.columns:
                text_col = col
                break
        
        if not text_col:
            return {"success": False, "error": "No complaint text column found"}
        
        # Analyze each complaint
        results = []
        for idx, row in df.iterrows():
            complaint_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            if complaint_text.strip():
                result = analyzer.analyze_complaint(complaint_text)
                results.append(result)
        
        # Generate insights
        category_counts = Counter(r.get("category", "Other") for r in results)
        priority_counts = Counter(r.get("priority", "Medium") for r in results)
        
        return {
            "success": True,
            "total_analyzed": len(results),
            "category_distribution": dict(category_counts),
            "priority_distribution": dict(priority_counts),
            "results": results[:20]  # Return first 20 results
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/statistics")
async def get_statistics():
    """Get complaint statistics"""
    try:
        stats = analyzer.get_statistics()
        return {"success": True, **stats}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/trends")
async def get_trends():
    """Get complaint trends"""
    try:
        trends = analyzer.get_trends()
        return {"success": True, **trends}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/categories")
async def get_categories():
    """Get all available categories"""
    categories = analyzer.get_categories()
    return {"success": True, "categories": categories}

@app.post("/suggestions")
async def get_suggestions(category: str):
    """Get AI suggestions for a specific category"""
    suggestions = analyzer.get_category_suggestions(category)
    return {"success": True, "category": category, "suggestions": suggestions}

# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI Complaint Categorization & Suggestion System")
    print("=" * 60)
    print(f"📁 Data file: {analyzer.data_file}")
    print(f"📊 Total complaints loaded: {len(analyzer.complaints_df) if hasattr(analyzer, 'complaints_df') else 0}")
    print(f"🏠 Server: http://127.0.0.1:8003")
    print(f"📊 Dashboard: http://127.0.0.1:8003/dashboard")
    print(f"📚 API Docs: http://127.0.0.1:8003/docs")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8003, log_level="info")