# HMS Core - Detailed Warden Use Cases Documentation 

## USE CASE W-1: View & Manage Complaints 

**Name:** Complaint Management and Resolution System 

**Description:** The warden receives, reviews, categorizes, prioritizes, and resolves student complaints about hostel facilities, food quality, cleanliness, maintenance, security, and other issues. The system provides AI-assisted categorization, suggested solutions, priority assignment, status tracking, and performance analytics. 

**Actors:** Warden (Primary), Student (Secondary), Hostel Owner (Secondary), Maintenance Staff (Secondary), Backend Server (Secondary), AI Complaint Categorization Service (Secondary), Notification Service (Secondary) 

**Organization Benefits:** 

- Improves complaint resolution efficiency and speed 

- Provides accountability through documented responses 

- Enables data-driven hostel improvements 

- Reduces escalations through timely action 

- Increases student satisfaction 

- Provides performance metrics for warden evaluation 

- Identifies recurring issues for systemic fixes 

**Preconditions:** 

- Warden account active with appropriate permissions 

- Students have submitted complaints 

- Complaint management system operational 

- Internet connectivity required 

- Warden has access to maintenance resources 

- AI categorization service trained and running 

**Triggers:** 

- Student submits new complaint 

- Warden opens complaint management dashboard 

- High-priority complaint requires immediate attention 

- Complaint remains unresolved beyond SLA 

- Warden manually searches for specific complaint 

- System prompts: "5 pending complaints need attention" 

**Main Course:** 

**Step 1:** Warden (Mr. Rajesh Gupta) starts morning rounds at Sunshine Boys Hostel 

- **Step 2:** Opens HMS Core Business App on tablet at 9:00 AM 

**Step 3:** Dashboard shows complaint summary widget: 

``` 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

🔔 COMPLAINTS OVERVIEW 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

New Complaints: 8 (Unassigned) In Progress: 12 Pending Review: 3 Resolved (24h): 15 

⚠️� HIGH PRIORITY: 2 complaints Require immediate attention 

[View All Complaints] [Quick Actions] 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

``` 

**Step 4:** Warden taps "View All Complaints" 

**Step 5:** Complaint management screen opens: 

``` 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 📋 COMPLAINT MANAGEMENT Sunshine Boys Hostel 

FILTERS: 

[All] [New] [In Progress] [Resolved] [High Priority] [Medium] [Low] 

# CATEGORIES: 

[All] [Food] [Cleanliness] [Maintenance] [Electricity] [WiFi] [Security] [Other] 

SORT BY: [Newest First ▼] 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

🔔 HIGH PRIORITY (2) 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

#C-2026-0234 | Room 301 

🔔 Maintenance | Submitted 2 hours ago 

Water not coming in bathroom since morning Need urgent fix, cannot use toilet or bath 

Student: Arjun Singh | ⏰ SLA: 4h remaining Images: [2 photos attached] AI Category: Plumbing Emergency Suggested Action: Call plumber immediately 

[ASSIGN TO PLUMBER] [MARK URGENT] [VIEW DETAILS] [RESPOND] 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

#C-2026-0235 | Room 405 Food Quality | Submitted 4 hours ago 

Food was spoiled in today's lunch, multiple students complained of stomach issues 

Student: Rahul Kumar | ⏰ SLA: 2h remaining Images: [3 photos of food] AI Category: Health & Safety - Critical Suggested Action: Investigate kitchen, check food storage, possible food poisoning 

[INVESTIGATE NOW] [CALL DOCTOR] [VIEW DETAILS] [RESPOND] 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

🟡 MEDIUM PRIORITY (6) 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

#C-2026-0236 | Room 208 

🔔 Cleanliness | Submitted 6 hours ago 

Common washroom on 2nd floor not cleaned for 2 days, very dirty and smelly 

Student: Amit Sharma | ⏰ SLA: 18h AI Category: Housekeeping Suggested Action: Schedule immediate cleaning 

[ASSIGN CLEANER] [VIEW DETAILS] 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

[Load More...] [Export Report] 

``` 

- **Step 6:** Warden identifies highest priority - food complaint **Step 7:** Taps on complaint #C-2026-0235 to view full details **Step 8:** Detailed complaint view opens: 

``` 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

COMPLAINT #C-2026-0235 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

STATUS: 🔔 NEW (High Priority) CATEGORY: Food Quality SUBCATEGORY: Food Spoilage/Poisoning 

SUBMITTED BY: Rahul Kumar (Room 405, Bed B) Contact: +91 98765 11111 

SUBMITTED: March 6, 2026, 5:15 AM SLA DEADLINE: March 6, 2026, 9:15 AM ⏰ TIME REMAINING: 2 hours 3 minutes 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

# COMPLAINT DESCRIPTION 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

"Yesterday's lunch had spoiled rice and dal. The smell was off but mess staff served it anyway. I ate some before realizing. Now having stomach pain and nausea since night. My roommate also sick. Other students from our floor also complaining. This is serious health risk. Please investigate and take action." 

# EVIDENCE ATTACHED: 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

[📷 Photo 1: Plate of rice showing discoloration] [📷 Photo 2: Dal container with unusual appearance] 

[📷 Photo 3: Multiple students in mess looking unwell] 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

# AI ANALYSIS 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

- 🔔 Category: Health & Safety - Critical 

- ⚠️� Risk Level: HIGH 

- 🔔 Affected Students: Potentially 15+ (floor 4) 

- ⏰ Urgency: IMMEDIATE 

# SUGGESTED ACTIONS: 

1. Verify with other students immediately 

2. Check kitchen food storage & hygiene 

3. Arrange medical check for affected students 

4. Investigate cook/mess staff negligence 

5. Quarantine remaining food samples 

6. Review food safety protocols 

# SIMILAR PAST COMPLAINTS: 

- #C-2026-0198 (2 weeks ago): "Food quality poor" 

- #C-2026-0176 (3 weeks ago): "Dal tasted odd" 

⚠️� PATTERN DETECTED: Recurring food quality issues. Systemic problem likely. 

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

# WARDEN ACTIONS 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

- [🔔 ARRANGE MEDICAL HELP] 

- [ INVESTIGATE KITCHEN NOW] 

- [🔔 CALL STUDENT] 

- [🔔 CHECK OTHER STUDENTS] 

- [🔔 COLLECT MORE EVIDENCE] 

- [ ESCALATE TO OWNER]⚡ 

- [🔔 RESPOND TO STUDENT] 

- [⏰ MARK AS IN PROGRESS] 

- [⏰ REJECT/MARK INVALID] 

RESPONSE TEMPLATE: [I understand this is serious. Taking immediate action to investigate...] 

# [Custom Response] 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

``` 

- **Step 9:** Warden realizes this is critical health issue **Step 10:** Immediately taps "INVESTIGATE KITCHEN NOW" **Step 11:** System creates investigation task: 

``` 

# ⏰ INVESTIGATION INITIATED 

Task: Kitchen Food Safety Investigation Priority: URGENT Assigned to: Warden (You) Status: In Progress 

Checklist: ☐ Visit kitchen immediately ☐ Check food storage temperatures ☐ Inspect refrigeration ☐ Interview cook/staff ☐ Collect food samples if available ☐ Document findings with photos ☐ Check affected students count 

[START INVESTIGATION] [Add Notes] ``` 

**Step 12:** Warden marks complaint "In Progress" and adds response: 

``` 

RESPONSE TO STUDENT: 

"Dear Rahul, 

Thank you for reporting this immediately. This is a serious matter and I am taking action right now. 

IMMEDIATE ACTIONS: 

- ⏰ Investigating kitchen and food storage 

- ⏰ Arranging medical check for you and 

affected students 

- ⏰ Cook being questioned 

- ⏰ Food samples being preserved 

I will personally visit your room within 30 minutes to check on your condition. 

Please let me know if symptoms worsen. Hostel has arrangement with nearby clinic for emergency. 

- Warden Rajesh Gupta Contact: +91 98765 00000" 

[SEND RESPONSE] [Save Draft] 

``` 

**Step 13:** Warden sends response, student receives notification **Step 14:** Warden rushes to kitchen for immediate inspection **Step 15:** Takes photos of kitchen conditions, checks refrigerator temperatures **Step 16:** Discovers: Refrigerator not cooling properly, food stored improperly **Step 17:** Returns to office, updates complaint with findings: 

``` 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

COMPLAINT UPDATE #C-2026-0235 

# INVESTIGATION FINDINGS: 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

Date: March 6, 2026, 10:00 AM Investigated by: Warden Rajesh Gupta 

KITCHEN INSPECTION RESULTS: 

- ⏰ Main refrigerator malfunction detected 

- Temperature: 18°C (should be 4°C) 

- Food stored at unsafe temperature 

- ⏰ Food storage protocols violated 

- Cooked food kept beyond safe time 

- Poor hygiene practices observed 

- ⏰ Kitchen cleanliness: Acceptable 

- ⏰ Staff cooperation: Good 

AFFECTED STUDENTS: Confirmed cases: 12 students Symptoms: Stomach pain, nausea, diarrhea Medical check arranged: Yes (Dr. Sharma visiting) 

# ROOT CAUSE: 

Refrigerator breakdown + staff negligence Food poisoning due to bacterial growth 

# IMMEDIATE ACTIONS TAKEN: 

- ⏰ Refrigerator repairman called (ETA 2 hours) 

- ⏰ All stored food discarded 

- ⏰ Doctor arranged for student checkups 

- ⏰ Cook suspended pending investigation 

- ⏰ New meal being prepared fresh 

- ⏰ Parents of affected students notified 

# EVIDENCE COLLECTED: 

[📷 6 photos of refrigerator, thermometer, food] [🔔 Written statement from cook] [📋 Refrigerator maintenance log] 

[UPDATE COMPLAINT] [Notify Management] ``` 

**Step 18:** Warden updates complaint status to "Being Resolved" **Step 19:** Escalates to hostel owner with full report 

**Step 20:** Arranges medical checkups for all 12 affected students **Step 21:** Calls refrigerator repair service - technician arrives **Step 22:** Fresh lunch prepared under warden's supervision **Step 23:** By evening, all students medically cleared, recovering 

**Step 24:** Warden marks complaint as "Resolved" with final update: 

# ``` 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ COMPLAINT #C-2026-0235 - RESOLVED ⏰ 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

# RESOLUTION SUMMARY: 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

Reported: March 6, 5:15 AM Resolved: March 6, 6:30 PM Resolution Time: 13 hours 15 minutes 

# PROBLEM: 

Food poisoning due to refrigerator malfunction and staff negligence 

# RESOLUTION ACTIONS: 

- ⏰ Refrigerator repaired (cooling restored) 

- ⏰ 12 affected students given medical care 

- ⏰ All students recovered, no hospitalizations 

- ⏰ Spoiled food discarded, kitchen sanitized 

- ⏰ Cook suspended for 1 week (training required) 

- ⏰ New food safety protocols implemented 

- ⏰ Daily temperature checks mandated 

# PREVENTIVE MEASURES: 

1. Weekly refrigerator temperature monitoring 

2. Backup refrigerator purchased 

3. Food safety training for all kitchen staff 

4. Surprise kitchen inspections scheduled 

5. Student feedback system for meals 

# STUDENT SATISFACTION: 

Rahul Kumar: "Thank you for quick action. Feeling better now. Appreciate the response." `Rating:` ⭐⭐⭐⭐⭐ `(5/5)` 

- TOTAL COST: 8,500₹ - Refrigerator repair: 3,500₹ - Medical consultations: 5,000₹ 

# LESSONS LEARNED: 

Need better preventive maintenance schedule for critical equipment like refrigerators. 

[CLOSE COMPLAINT] [Generate Report] [Share with Management] 

``` 

**Step 25:** Complaint marked resolved, all stakeholders notified 

**Step 26:** System generates performance metrics for this complaint 

**Step 27:** Warden continues with other pending complaints 

--- 

**Alternate Course:** 

- **AC1: Complaint Requires External Vendor** 

- *Condition: Plumbing issue needs professional plumber* 

**Step 1:** Warden reviews water supply complaint (#C-2026-0234) 

- **Step 2:** Realizes this needs professional plumber, not in-house fix **Step 3:** Taps "ASSIGN TO PLUMBER" button **Step 4:** System shows registered vendors: 

``` 

SELECT SERVICE PROVIDER 

# 🔔 PLUMBING SERVICES 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

○ Quick Fix Plumbers `Rating:` ⭐⭐⭐⭐ `(4.2/5)` Last used: 2 weeks ago Response time: 1-2 hours Cost: ₹₹ (Moderate) Contact: +91 98765 22222 [SELECT] 

● Sharma Plumbing Services (Recommended) `Rating:` ⭐⭐⭐⭐⭐ `(4.8/5)` Last used: 3 days ago Response time: 30-60 mins Cost: ₹₹₹ (Slightly higher) Contact: +91 98765 33333 [SELECT] ✓ Selected 

○ Budget Plumbing Co. `Rating:` ⭐⭐⭐ `(3.5/5)` Last used: 1 month ago Response time: 2-3 hours Cost:  (Low)₹ Contact: +91 98765 44444 [SELECT] 

# [CALL SELECTED VENDOR] [REQUEST QUOTE] 

``` 

**Step 5:** Warden selects Sharma Plumbing (best rated, fast) 

- **Step 6:** Taps "CALL SELECTED VENDOR" 

- **Step 7:** System initiates call, auto-fills work order details 

- **Step 8:** Plumber agrees to visit in 1 hour 

- **Step 9:** Warden updates complaint: 

``` 

# UPDATE: Plumber called 

Vendor: Sharma Plumbing Services Expected arrival: 11:00 AM Work order: #WO-2026-0156 Estimated cost: 1,500-2,000₹ 

Student notified of timeline. 

Status: In Progress → Awaiting Vendor 

``` 

**Step 10:** When plumber arrives, warden supervises work 

- **Step 11:** Work completed, warden marks resolved with bill photo 

--- 

**AC2: Complaint is Duplicate/Invalid** 

*Condition: Student complains about already-resolved issue* 

- **Step 1:** Warden reviews complaint: "WiFi not working in Room 301" 

- **Step 2:** Checks records - WiFi router replaced yesterday for Room 301 **Step 3:** Calls student to verify 

- **Step 4:** Student confirms: "Oh yes, it's working now. I forgot to update." 

- **Step 5:** Warden marks complaint as "Invalid - Already Resolved" 

- **Step 6:** Adds closing note: 

``` 

# RESOLUTION: Complaint Invalid 

This issue was already resolved on March 5 when WiFi router was replaced for Room 301. 

Student confirmed WiFi working properly now. Complaint submitted before resolution was tested by student. 

Action: None required Status: Closed (Invalid) Reason: Duplicate/Already Resolved 

# [CLOSE COMPLAINT] 

``` 

**Step 7:** Student receives polite notification explaining closure 

--- 

- **AC3: Complaint Escalation to Owner** 

- *Condition: Major issue beyond warden's authority* 

- **Step 1:** Student complains: "No hot water for 3 days, geyser broken" 

- **Step 2:** Warden investigates: Central heating system failed 

- **Step 3:** Realizes repair cost 50,000+ (beyond warden budget authority)₹ 

- **Step 4:** Taps "ESCALATE TO OWNER" button 

- **Step 5:** Escalation form appears: 

``` 

# ESCALATE COMPLAINT TO OWNER 

Complaint: #C-2026-0240 Issue: Central heating system failure Estimated Cost: 50,000-75,000₹ 

# ESCALATION REASON: 

- Exceeds warden budget authority (₹25,000) 

- Requires policy decision 

- Legal/safety concern 

- Student demanding owner intervention 

# URGENCY: 

- HIGH - Affects all students (85) 

- Medium - Affects one floor/wing 

- Low - Individual room issue 

WARDEN'S RECOMMENDATION: [Immediate repair required. Winter season, students need hot water for bathing. Affecting all 85 students. Request emergency approval for heating system replacement. Alternative: Install individual geysers in bathrooms (long-term solution).] 

ESTIMATED TIMELINE: 

Repair: 3-5 days | Cost: 50,000₹ OR Install geysers: 1 week | Cost: 75,000₹ 

# [SEND TO OWNER] [Save Draft] [Cancel] 

``` 

**Step 6:** Warden sends escalation to owner 

- **Step 7:** Owner receives notification with full details 

- **Step 8:** Owner approves within 2 hours: "Proceed with heating repair" 

**Step 9:** Warden gets approval notification, proceeds with contractor **Step 10:** Updates complaint: "Escalated → Approved → In Progress" 

--- 

**AC4: Student Disputes Resolution** 

*Condition: Student unhappy with how complaint resolved* 

- **Step 1:** Warden marks complaint resolved: "Room cleaned" 

- **Step 2:** Student reopens: "Room still dirty, not properly cleaned" 

- **Step 3:** Warden receives notification: 

``` 

# 🔔 COMPLAINT REOPENED 

#C-2026-0238 | Room Cleanliness Previously: Resolved 

Now: Reopened by student 

STUDENT'S REASON: "Room was superficially cleaned. Bathroom still has mold, corners have dust. Cleaning was rushed and incomplete. Please inspect personally before marking resolved." 

[REINVESTIGATE] [CALL STUDENT] [VIEW ORIGINAL RESOLUTION] [DISCUSS] ``` 

- **Step 4:** Warden personally visits room to verify 

- **Step 5:** Finds student is correct - cleaning was inadequate 

- **Step 6:** Calls cleaning staff, redoes cleaning under supervision 

- **Step 7:** Takes photos of properly cleaned room 

- **Step 8:** Updates complaint: 

``` 

# COMPLAINT RE-RESOLVED 

Inspection Date: March 6, 2026 Warden: Rajesh Gupta 

# FINDINGS: 

Student complaint valid. Initial cleaning was inadequate. Bathroom mold and corners were missed. 

# CORRECTIVE ACTION: 

- ⏰ Re-cleaning done under warden supervision 

- ⏰ Bathroom mold removed with cleaner 

- ⏰ All corners and hard-to-reach areas cleaned 

- ⏰ Photos taken for verification 

- ⏰ Cleaning staff counseled on quality 

# QUALITY ASSURANCE: [📷 4 photos showing properly cleaned room] 

Student invited to inspect and confirm. 

Apology to student for initial oversight. 

Status: Reopened → Re-Resolved Resolution Quality: Verified by Warden 

[CLOSE COMPLAINT] [Request Student Confirmation] ``` 

**Step 9:** Student inspects, confirms satisfaction, rates 4/5 

**Step 10:** Complaint finally closed with lessons learned 

--- 

**AC5: Bulk Complaint Processing** 

*Condition: Multiple students complain about same issue* 

**Step 1:** Warden receives 8 separate complaints: "No electricity in West Wing" **Step 2:** System detects pattern, groups complaints: 

``` 

# ⚡ BULK COMPLAINT DETECTED 

# 8 RELATED COMPLAINTS GROUPED 

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

Issue: Power outage - West Wing Affected: Rooms 401-408 (16 students) Time: Since 8:00 PM 

GROUPED COMPLAINTS: 

#C-2026-0241 - Amit (Room 401) #C-2026-0242 - Rohit (Room 402) #C-2026-0243 - Suresh (Room 403) ... 5 more 

ROOT CAUSE (Suggested): Circuit breaker trip or transformer issue in West Wing power distribution 

# RECOMMENDED ACTION: 

Single fix will resolve all 8 complaints 

[ASSIGN TO ELECTRICIAN] [NOTIFY ALL STUDENTS] [INVESTIGATE NOW] ``` 

**Step 3:** Warden calls electrician, identifies blown fuse 

**Step 4:** Electrician replaces fuse, power restored 

**Step 5:** Warden resolves all 8 complaints with single update: 

``` 

BULK RESOLUTION 

Issue: West Wing Power Outage Complaints Resolved: 8 (all) 

ROOT CAUSE: Fuse blown in West Wing panel Due to: Power surge during evening 

RESOLUTION: 

- ⏰ Fuse replaced 

- ⏰ Power restored at 9:30 PM 

- ⏰ All 16 students affected now have electricity 

# PREVENTIVE ACTION: 

Voltage stabilizer to be installed to prevent future power surges. 

All 8 students notified of resolution. 

# [CLOSE ALL COMPLAINTS] [Send Mass Update] 

``` 

**Step 6:** All 8 students receive resolution notification simultaneously 

--- 

**Exception Courses:** 

**EX1: Complaint Submitted During Warden Off-Hours** 

*Condition: Emergency complaint at 2:00 AM* 

- **Step 1:** Student submits urgent complaint at 2:00 AM 

- **Step 2:** System detects warden offline (sleeping) 

- **Step 3:** AI analyzes severity using keywords 

- **Step 4:** If critical (fire, medical, security): 

- Auto-escalates to hostel owner 

- Sends emergency alert to warden's phone (overrides silent mode) 

- Notifies backup emergency contact 

**Step 5:** If non-critical: 

- Queues complaint for next morning 

- Sends auto-response to student: 

``` 

COMPLAINT RECEIVED 

Thank you for submitting your complaint. 

Your complaint has been registered and will be reviewed by the warden by 9:00 AM tomorrow. 

For emergencies, please call: Warden: +91 98765 00000 Emergency: +91 98765 99999 

Complaint ID: #C-2026-0245 Expected response: Within 12 hours 

- HMS Automated System 

``` 

**Step 6:** Warden sees queued complaint at 9 AM, takes action 

--- 

- **EX2: Too Many Complaints Overwhelming System** 

- *Condition: 50+ complaints submitted in one day* 

- **Step 1:** System detects abnormal volume (50 complaints vs usual 10) **Step 2:** Alerts warden: 

``` 

# ⚠️� HIGH COMPLAINT VOLUME ALERT 

50 new complaints today (500% above normal) 

Possible Reasons: 

- Systemic issue affecting multiple students 

- Seasonal issue (monsoon, winter, exams) 

- Service disruption 

# BREAKDOWN BY CATEGORY: 

Food Quality: 30 complaints 

- 🔔 Cleanliness: 10 complaints 

- 🔔 Electricity: 5 complaints `🌐` WiFi: 3 complaints 

Other: 2 complaints 

# RECOMMENDATION: 

Focus on Food Quality (60% of complaints) Likely systemic issue in kitchen. 

[PRIORITIZE COMPLAINTS] [ESCALATE TO OWNER] [ANALYZE PATTERNS] [REQUEST HELP] 

``` 

- **Step 3:** Warden prioritizes highest-impact issues first 

- **Step 4:** Requests assistant warden or owner's help for backlog 

**Step 5:** Addresses systemic food issue, resolves 30 complaints at once 

--- 

- **EX3: Complaint Evidence Missing/Unclear** 

- *Condition: Student complaint too vague* 

- **Step 1:** Complaint: "Room problem, fix it" 

- **Step 2:** Warden confused - what problem? 

- **Step 3:** System prompts: 

``` 

# ⚠️� INCOMPLETE COMPLAINT 

This complaint lacks sufficient detail for resolution. 

# MISSING INFORMATION: 

- Specific issue description 

- Location details 

- Photos/evidence 

- Urgency level 

# ACTIONS: 

[REQUEST MORE DETAILS FROM STUDENT] [CALL STUDENT FOR CLARIFICATION] [VISIT ROOM TO INSPECT] [MARK AS INCOMPLETE] ``` 

**Step 4:** Warden sends clarification request: 

``` 

# CLARIFICATION NEEDED 

Dear Student, 

Thank you for your complaint. To help you better, I need more information: 

1. What specific problem in your room? (Plumbing, electrical, furniture, etc.) 

2. When did this problem start? 

3. Is this urgent or can wait 24 hours? 

4. Can you share photos if applicable? 

Please update complaint with details so I can take appropriate action. 

- Warden Rajesh Gupta 

``` 

**Step 5:** Student provides details, warden proceeds 

**Step 6:** If student doesn't respond in 48 hours, auto-closes with note 

--- 

**EX4: Warden Unavailable for Extended Period** 

*Condition: Warden on medical leave for 1 week* 

**Step 1:** Warden falls sick, applies leave in system 

**Step 2:** System prompts: 

``` 

LEAVE APPLICATION - WARDEN 

Leave Period: March 7-14, 2026 (7 days) Reason: Medical leave 

COMPLAINT MANAGEMENT: 

Who will handle pending complaints? 

- Auto-assign to Assistant Warden 

- Escalate all to Hostel Owner 

- Delegate to: [Temporary Warden Name] 

# NOTIFICATION: 

- ⏰ Notify all students of temporary warden 

- ⏰ Update emergency contact numbers 

- ⏰ Forward pending complaints (18) 

# [CONFIRM LEAVE] [Cancel] 

``` 

- **Step 3:** Complaints automatically reassigned to temporary warden 

- **Step 4:** Students notified of change 

- **Step 5:** When original warden returns, gets summary of actions taken 

--- 

**Postconditions:** 

**Success:** 

- ⏰ All complaints reviewed within SLA timelines 

- ⏰ Students receive timely responses and updates 

- ⏰ Issues resolved with documented evidence 

- ⏰ Quality of resolution verified 

- ⏰ Patterns identified for systemic improvements 

- ⏰ Performance metrics tracked (resolution time, satisfaction) 

- ⏰ Hostel conditions improved through feedback loop 

- ⏰ Analytics logged: complaint_volume, resolution_rate, student_satisfaction, category_trends 

**Failure:** 

- ⏰ Escalation process activated for unresolved complaints 

- ⏰ Owner/management notified of warden performance issues 

- ⏰ Students can appeal or reopen complaints 

- ⏰ Backup warden assigned if primary overwhelmed 

- ⏰ System flags recurring issues for attention 

--- 

## USE CASE W-2: View All Students' Attendance 

**Name:** Student Attendance Monitoring and Reporting System 

**Description:** The warden monitors real-time attendance of all hostel students, tracks entry/exit patterns, identifies irregularities, enforces curfew policies, generates attendance reports, and ensures student safety through movement tracking. The system provides dashboard views, alerts for policy violations, analytics on attendance trends, and parent notification integration. 

**Actors:** Warden (Primary), Student (Secondary), Biometric/RFID System (Secondary), Parent (Secondary), Hostel Owner (Secondary), Backend Server (Secondary), Notification Service (Secondary), Analytics Engine (Secondary) 

**Organization Benefits:** 

- Ensures student safety through movement tracking 

- Enables enforcement of hostel policies and curfews 

- Provides accountability for student whereabouts 

- Reduces unauthorized absences and late returns 

- Generates data for parent communication 

- Identifies at-risk students (irregular patterns) 

- Provides attendance analytics for management decisions 

**Preconditions:** 

- Warden has active HMS Core Business App account 

- Hostel has entry/exit tracking system (biometric/RFID/manual) 

- Students have registered biometric/RFID credentials 

- Attendance tracking system integrated with HMS Core 

- Internet connectivity required for real-time updates 

- Student-parent relationships configured for notifications 

**Triggers:** 

- Warden opens attendance dashboard 

- Student enters or exits hostel premises 

- Curfew time reached with students still outside 

- Attendance report request from owner/parent 

- Irregular attendance pattern detected 

- System prompts: "3 students haven't returned by curfew" 

# **Main Course:** 

**Step 1:** Warden (Mr. Rajesh Gupta) starts evening shift at 6:00 PM **Step 2:** Opens HMS Core Business App on tablet 

- **Step 3:** Taps "Attendance Dashboard" from main menu 

**Step 4:** Real-time attendance overview screen loads: 

``` 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

📊 ATTENDANCE DASHBOARD Sunshine Boys Hostel | March 6, 2026 

CURRENT STATUS (6:00 PM) 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 

Total Students: 85 

- ⏰ Inside Hostel: 52 (61%) 

🔔 Outside Hostel: 33 (39%) 

⚠️� Unknown Status: 0 (0%) 

CURFEW COMPLIANCE 

━━━━━━━━━━━━━━━━━━━━━━ 

