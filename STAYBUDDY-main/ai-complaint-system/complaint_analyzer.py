# ai-complaint-system/complaint_analyzer.py
# ══════════════════════════════════════════════════════════════════════════════
# CORE COMPLAINT ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional
import json
import re
from pathlib import Path
import hashlib

class ComplaintAnalyzer:
    """AI-powered complaint analyzer with categorization and suggestion capabilities"""
    
    def __init__(self, data_path: str = "data/complaints.csv"):
        self.data_path = Path(data_path)
        self.complaints_df = None
        self.knowledge_base = self._load_knowledge_base()
        self.load_data()
        
    def _load_knowledge_base(self) -> Dict:
        """Load the complaint knowledge base"""
        return {
            "Cleanliness": {
                "keywords": ["dirty", "clean", "cleaning", "dust", "dustbin", "bathroom", "toilet", 
                            "garbage", "trash", "sweep", "floor", "stain", "smell", "odour", 
                            "mosquito", "cockroach", "pest", "insect", "rat", "hygiene", "unhygienic",
                            "filthy", "mess", "waste", "bin"],
                "subcategories": ["Pest Control", "Room Cleaning", "Bathroom", "Garbage", "Common Area"],
                "suggested_actions": [
                    "Schedule immediate housekeeping",
                    "Arrange pest control inspection",
                    "Add area to daily cleaning checklist",
                    "Review cleaning staff schedule",
                    "Install air fresheners"
                ],
                "resolution_time": "1-2 days",
                "priority_boost": 0,
                "color": "#0A6B6E"
            },
            "Maintenance": {
                "keywords": ["broken", "repair", "fix", "leak", "water", "pipe", "electricity", 
                            "fan", "light", "bulb", "switch", "door", "lock", "window", "crack",
                            "geyser", "heater", "ac", "wifi", "internet", "power", "outage",
                            "not working", "damaged", "fault"],
                "subcategories": ["Electrical", "Plumbing", "WiFi/Internet", "Appliances", "Structural"],
                "suggested_actions": [
                    "Assign maintenance technician within 24 hours",
                    "Log issue in maintenance register",
                    "Contact relevant contractor",
                    "Provide temporary solution while pending",
                    "Escalate to owner if not resolved in 48 hours"
                ],
                "resolution_time": "1-3 days",
                "priority_boost": 1,
                "color": "#B85C00"
            },
            "Food Quality": {
                "keywords": ["food", "meal", "lunch", "dinner", "breakfast", "taste", "stale", 
                            "rotten", "cold", "dirty", "hair", "quantity", "portion", "small",
                            "quality", "bad", "horrible", "cook", "kitchen", "menu", "undercooked"],
                "subcategories": ["Taste/Quality", "Hygiene", "Portion Size", "Availability"],
                "suggested_actions": [
                    "Inspect kitchen and food storage immediately",
                    "Check food quality with cook and supplier",
                    "Review meal plan and portion sizes",
                    "Conduct hygiene audit of kitchen area",
                    "Warn cook; escalate if repeated"
                ],
                "resolution_time": "Same day",
                "priority_boost": 0,
                "color": "#2D6A11"
            },
            "Safety/Security": {
                "keywords": ["theft", "stolen", "missing", "security", "guard", "cctv", 
                            "unauthorized", "stranger", "unsafe", "dangerous", "harass", 
                            "emergency", "fire", "blocked", "exit", "injury", "attack"],
                "subcategories": ["Theft", "Unauthorized Entry", "Security Guard", "Emergency"],
                "suggested_actions": [
                    "Investigate immediately and file incident report",
                    "Review CCTV footage from reported time",
                    "Involve police if serious safety issue",
                    "Increase security patrols in reported area",
                    "Install additional security measures"
                ],
                "resolution_time": "Immediate - 24 hours",
                "priority_boost": 2,
                "color": "#C0392B"
            },
            "Noise/Disturbance": {
                "keywords": ["noise", "loud", "music", "shouting", "screaming", "party", 
                            "sleep", "quiet", "disturb", "night", "late", "midnight"],
                "subcategories": ["Night Noise", "Party Noise", "Roommate Noise", "External Noise"],
                "suggested_actions": [
                    "Warn the noisy party and issue formal notice",
                    "Enforce quiet hours policy (10 PM - 7 AM)",
                    "Mediate between affected students",
                    "Issue fine if noise policy violated repeatedly",
                    "Consider room reassignment if persistent"
                ],
                "resolution_time": "Same day",
                "priority_boost": 0,
                "color": "#7B3FC4"
            },
            "Staff Behavior": {
                "keywords": ["warden", "staff", "cook", "guard", "cleaner", "rude", "behavior",
                            "disrespect", "shout", "threat", "unfair", "biased", "abuse", 
                            "unprofessional", "misbehave", "attitude"],
                "subcategories": ["Warden", "Staff", "Security", "Cleaner"],
                "suggested_actions": [
                    "Counsel the staff member formally",
                    "Document incident with date, time and witnesses",
                    "Involve hostel owner if behavior is repeated",
                    "Issue written warning to staff member",
                    "Arrange conflict resolution training"
                ],
                "resolution_time": "1-2 days",
                "priority_boost": 1,
                "color": "#185FA5"
            },
            "Roommate Issue": {
                "keywords": ["roommate", "sharing", "personal", "space", "belongings", 
                            "borrow", "smoke", "fight", "conflict", "argument", "privacy", 
                            "guest", "visitor", "smoking"],
                "subcategories": ["Personal Space", "Habits", "Conflict", "Guests"],
                "suggested_actions": [
                    "Mediate between roommates with warden present",
                    "Set clear room rules and document agreement",
                    "Offer room change if mediation fails",
                    "Warn student about hostel conduct policy",
                    "Arrange periodic check-ins"
                ],
                "resolution_time": "2-3 days",
                "priority_boost": 0,
                "color": "#E91E8C"
            },
            "Billing/Payment": {
                "keywords": ["bill", "billing", "charge", "money", "fee", "payment", "rent",
                            "extra", "refund", "deposit", "receipt", "invoice", "overcharge",
                            "deduction", "fine", "penalty"],
                "subcategories": ["Overcharge", "Refund", "Payment", "Deposit"],
                "suggested_actions": [
                    "Review fee structure and provide itemized receipt",
                    "Cross-check payment records with management",
                    "Process refund if charge was incorrect",
                    "Escalate billing dispute to owner",
                    "Update billing system to prevent recurrence"
                ],
                "resolution_time": "3-5 days",
                "priority_boost": 0,
                "color": "#B85C00"
            }
        }
    
    def load_data(self):
        """Load complaints data from CSV"""
        try:
            if self.data_path.exists():
                self.complaints_df = pd.read_csv(self.data_path)
                print(f"✅ Loaded {len(self.complaints_df)} complaints from {self.data_path}")
            else:
                print(f"⚠️ Data file not found: {self.data_path}")
                self.complaints_df = pd.DataFrame()
        except Exception as e:
            print(f"⚠️ Error loading data: {e}")
            self.complaints_df = pd.DataFrame()
    
    def _categorize_complaint(self, text: str) -> Dict[str, Any]:
        """Categorize a single complaint"""
        text_lower = text.lower()
        
        # Calculate scores for each category
        scores = {}
        matched_keywords = {}
        
        for category, data in self.knowledge_base.items():
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
        
        # Determine category
        if not scores:
            category = "Other"
            matched = []
            confidence = 0.5
        else:
            category = max(scores, key=scores.get)
            matched = matched_keywords.get(category, [])
            max_score = scores[category]
            total_score = sum(scores.values())
            confidence = min(0.95, max_score / total_score) if total_score > 0 else 0.7
        
        # Find subcategory
        category_data = self.knowledge_base.get(category, {})
        subcategory = "General"
        best_score = 0
        for sub in category_data.get("subcategories", []):
            sub_lower = sub.lower()
            if sub_lower in text_lower:
                score = 1
                if score > best_score:
                    best_score = score
                    subcategory = sub
        
        # Determine priority
        high_priority = ["urgent", "immediately", "emergency", "dangerous", "stolen", 
                        "missing", "fire", "injury", "blocked", "theft", "broken"]
        low_priority = ["small", "minor", "little", "slightly", "prefer", "suggest"]
        
        if any(word in text_lower for word in high_priority) or category_data.get("priority_boost", 0) >= 2:
            priority = "High"
        elif any(word in text_lower for word in low_priority):
            priority = "Low"
        else:
            priority = "Medium"
        
        return {
            "category": category,
            "subcategory": subcategory,
            "priority": priority,
            "confidence": confidence,
            "matched_keywords": matched[:5],
            "color": category_data.get("color", "#5F5E5A")
        }
    
    def _generate_suggestion(self, category: str, subcategory: str, priority: str, text: str) -> str:
        """Generate AI suggestion based on complaint details"""
        category_data = self.knowledge_base.get(category, {})
        text_lower = text.lower()
        
        # Priority-based suggestions
        if priority == "High":
            urgent_suggestions = {
                "Safety/Security": "URGENT: Investigate immediately! Involve security and management. File incident report and review CCTV.",
                "Maintenance": "URGENT: Dispatch maintenance team immediately. This is a critical issue requiring same-day resolution.",
                "Food Quality": "URGENT: Inspect kitchen now! Stop food service if health hazard. Contact health inspector if needed."
            }
            if category in urgent_suggestions:
                return urgent_suggestions[category]
        
        # Category-specific suggestions
        base_suggestions = {
            "Cleanliness": f"Schedule immediate housekeeping for {subcategory.lower()}. Conduct spot inspection within 24 hours. Add to daily checklist.",
            "Maintenance": f"Assign technician for {subcategory.lower()} repair within 24 hours. Log in maintenance system. Update student on progress.",
            "Food Quality": f"Inspect kitchen immediately. Check {subcategory.lower()} issues. Review quality control measures. Take corrective action.",
            "Safety/Security": f"Investigate {subcategory.lower()} issue. Review security protocols. Take preventive measures. Document incident.",
            "Noise/Disturbance": f"Address {subcategory.lower()} complaint. Enforce quiet hours. Mediate if needed. Issue warning if repeated.",
            "Staff Behavior": f"Counsel staff about professional behavior. Document incident. Monitor improvement. Escalate if pattern continues.",
            "Roommate Issue": f"Mediate {subcategory.lower()} conflict. Set clear room rules. Consider room change if unresolved.",
            "Billing/Payment": f"Review {subcategory.lower()} charges. Provide detailed breakdown. Process adjustment if error found."
        }
        
        suggestion = base_suggestions.get(category, f"Review {category} complaint. Respond within 48 hours with action plan.")
        
        # Enhance with specific keywords from complaint
        if "pest" in text_lower or "mosquito" in text_lower:
            suggestion += " Schedule pest control immediately."
        if "water" in text_lower and "leak" in text_lower:
            suggestion += " Check for water damage and mold."
        if "theft" in text_lower or "stolen" in text_lower:
            suggestion += " File police report if valuables stolen."
        
        return suggestion
    
    def _find_similar_complaints(self, text: str, limit: int = 3) -> List[Dict]:
        """Find similar past complaints"""
        if self.complaints_df.empty:
            return []
        
        text_lower = text.lower()
        similar = []
        
        # Simple keyword matching for similarity
        complaint_col = None
        for col in ['complaint_text', 'text', 'complaint', 'description']:
            if col in self.complaints_df.columns:
                complaint_col = col
                break
        
        if not complaint_col:
            return []
        
        for idx, row in self.complaints_df.iterrows():
            complaint_text = str(row[complaint_col]) if pd.notna(row[complaint_col]) else ""
            if complaint_text:
                # Calculate simple similarity based on common words
                words1 = set(text_lower.split())
                words2 = set(complaint_text.lower().split())
                common = len(words1 & words2)
                similarity = common / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0
                
                if similarity > 0.2 and complaint_text != text:
                    similar.append({
                        "text": complaint_text[:150] + "...",
                        "similarity": round(similarity, 2),
                        "category": row.get("category", "Unknown") if "category" in row.columns else "Unknown",
                        "resolution": row.get("status", "Not specified") if "status" in row.columns else "Unknown"
                    })
        
        similar.sort(key=lambda x: x["similarity"], reverse=True)
        return similar[:limit]
    
    def analyze_complaint(self, text: str) -> Dict[str, Any]:
        """Complete analysis of a complaint"""
        # Generate complaint ID
        complaint_id = f"CMP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Categorize
        category_info = self._categorize_complaint(text)
        
        # Generate suggestion
        suggestion = self._generate_suggestion(
            category_info["category"], 
            category_info["subcategory"],
            category_info["priority"],
            text
        )
        
        # Find similar complaints
        similar = self._find_similar_complaints(text)
        
        # Get suggested actions
        category_data = self.knowledge_base.get(category_info["category"], {})
        suggested_actions = category_data.get("suggested_actions", [
            "Review complaint and respond within 48 hours",
            "Discuss with student to understand full issue",
            "Escalate to management if needed"
        ])
        
        # Estimate resolution time
        resolution_time = category_data.get("resolution_time", "2-3 days")
        if category_info["priority"] == "High":
            resolution_time = "Same day / 24 hours"
        
        return {
            "complaint_id": complaint_id,
            "original_text": text,
            "category": category_info["category"],
            "subcategory": category_info["subcategory"],
            "priority": category_info["priority"],
            "suggestion": suggestion,
            "confidence": category_info["confidence"],
            "similar_complaints": similar,
            "suggested_actions": suggested_actions[:3],
            "estimated_resolution_time": resolution_time,
            "color": category_info["color"],
            "timestamp": datetime.now().isoformat()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about complaints"""
        if self.complaints_df.empty:
            return {
                "total_complaints": 0,
                "category_distribution": {},
                "priority_distribution": {},
                "status_distribution": {},
                "resolution_rate": 0
            }
        
        # Find relevant columns
        cat_col = None
        for col in ['category', 'Category']:
            if col in self.complaints_df.columns:
                cat_col = col
                break
        
        status_col = None
        for col in ['status', 'Status']:
            if col in self.complaints_df.columns:
                status_col = col
                break
        
        # Calculate distributions
        category_dist = {}
        if cat_col:
            category_dist = self.complaints_df[cat_col].value_counts().to_dict()
        
        # Calculate resolution rate
        resolution_rate = 0
        if status_col:
            resolved = self.complaints_df[status_col].str.contains('Resolved|Closed', case=False, na=False).sum()
            resolution_rate = round((resolved / len(self.complaints_df)) * 100, 1)
        
        return {
            "total_complaints": len(self.complaints_df),
            "category_distribution": category_dist,
            "priority_distribution": {"High": 0, "Medium": 0, "Low": 0},  # Would need priority column
            "status_distribution": self.complaints_df[status_col].value_counts().to_dict() if status_col else {},
            "resolution_rate": resolution_rate
        }
    
    def get_trends(self) -> Dict[str, Any]:
        """Get complaint trends"""
        recent = []
        if not self.complaints_df.empty:
            text_col = None
            for col in ['complaint_text', 'text', 'complaint', 'description']:
                if col in self.complaints_df.columns:
                    text_col = col
                    break
            
            if text_col:
                for idx, row in self.complaints_df.head(10).iterrows():
                    recent.append({
                        "complaint_text": str(row[text_col])[:200] if pd.notna(row[text_col]) else "",
                        "category": row.get("category", "Unknown") if "category" in self.complaints_df.columns else "Unknown",
                        "priority": row.get("priority", "Medium") if "priority" in self.complaints_df.columns else "Medium"
                    })
        
        return {
            "recent_complaints": recent,
            "total_trend": "Stable",
            "peak_hours": "Evening (6 PM - 9 PM)",
            "common_issues": ["Cleanliness", "Maintenance", "Noise"]
        }
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.knowledge_base.keys()) + ["Other"]
    
    def get_category_suggestions(self, category: str) -> List[str]:
        """Get suggestions for a specific category"""
        return self.knowledge_base.get(category, {}).get("suggested_actions", [
            "Review complaint and respond appropriately",
            "Document the issue for future reference",
            "Follow up with student for resolution"
        ])