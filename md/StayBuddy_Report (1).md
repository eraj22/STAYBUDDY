

<!-- Start of picture text -->
ance<br>ig "NZ<br>° Gy r\ ec<br>ey Sonn nS49<br>BP som cs<br><!-- End of picture text -->

# **Contents** 

|**List of**|**Figures**|**iii**|
|---|---|---|
|**List of**|**Tables**|**iv**|
|**1**<br>**Intr**|**oduction**|**1**|
|1.1|Background . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>1|
|1.2|Existing Solutions . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>1|
|1.3|Problem Statement<br>. . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>2|
|1.4|Scope<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>3|
|1.5|Modules . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>3|
||1.5.1<br>User Management Module . . . . . . . . . . . . . . . .|. . . . . . . .<br>3|
||1.5.2<br>Hostel Management Module . . . . . . . . . . . . . . .|. . . . . . . .<br>3|
||1.5.3<br>Booking Management Module . . . . . . . . . . . . . .|. . . . . . . .<br>4|
||1.5.4<br>Review and Rating Module . . . . . . . . . . . . . . . .|. . . . . . . .<br>4|
||1.5.5<br>Complaint Management Module . . . . . . . . . . . . .|. . . . . . . .<br>4|
||1.5.6<br>Recommendation Module<br>. . . . . . . . . . . . . . . .|. . . . . . . .<br>4|
||1.5.7<br>Communication and Notifcation Module<br>. . . . . . . .|. . . . . . . .<br>4|
|1.6|Work Division . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>5|
|**2**<br>**Pro**|**ject Requirements**|**6**|
|2.1|Use Case / Event Response Table / Storyboarding . . . . . . . .|. . . . . . . .<br>6|
||2.1.1<br>Use Case Diagram<br>. . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>6|
||2.1.2<br>Event Response Table. . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>7|
|2.2|Functional Requirements . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>7|
||2.2.1<br>User Management Module . . . . . . . . . . . . . . . .|. . . . . . . .<br>7|
||2.2.2<br>Hostel Management Module . . . . . . . . . . . . . . .|. . . . . . . .<br>7|
||2.2.3<br>Recommendation Module<br>. . . . . . . . . . . . . . . .|. . . . . . . .<br>8|
||2.2.4<br>Booking Module<br>. . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>8|
||2.2.5<br>Complaint and Chatbot Module<br>. . . . . . . . . . . . .|. . . . . . . .<br>8|
||2.2.6<br>Warden and Parent Modules<br>. . . . . . . . . . . . . . .|. . . . . . . .<br>8|
|2.3|Non-Functional Requirements<br>. . . . . . . . . . . . . . . . . .|. . . . . . . .<br>8|
||2.3.1<br>Usability<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . .<br>8|



i 

_StayBuddy – Smart Hostel Recommendation System_ 

_CONTENTS_ 

||2.3.2|Performance. . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>9|
|---|---|---|---|
||2.3.3|Security . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>9|
||2.3.4|Availability . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>9|
|**3**<br>**Syst**|**em Ov**|**erview**|**10**|
|3.1|Archit|ectural Design<br>. . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>10|
||3.1.1|Box and Line Diagram . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>10|
|3.2|Data|Design . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>11|
|3.3|Doma|in Model<br>. . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>11|
|3.4|Desig|n Models (Up to Current Iteration) . . . . . . . . . . . .|. . . . . . . . .<br>12|
||3.4.1|Activity Diagram . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>12|
||3.4.2|Class Diagram<br>. . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>15|
||3.4.3|Sequence Diagram<br>. . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>16|
||3.4.4|State Transition Diagram . . . . . . . . . . . . . . . .|. . . . . . . . .<br>16|
||3.4.5|Data Flow Diagram (DFD) . . . . . . . . . . . . . . .|. . . . . . . . .<br>17|
|||3.4.5.1<br>Context Diagram (Level 0) . . . . . . . . . .|. . . . . . . . .<br>17|
|||3.4.5.2<br>Level 1 DFD . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
|||3.4.5.3<br>Level 2 DFD . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>18|
|**4**<br>**Imp**|**lement**|**ation and Testing**|**20**|
|4.1|Algor|ithm Design . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>20|
||4.1.1|Hybrid Recommendation Algorithm . . . . . . . . . .|. . . . . . . . .<br>20|
||4.1.2|Intent Classifcation for Chatbot . . . . . . . . . . . .|. . . . . . . . .<br>20|
|4.2|Exter|nal APIs and SDKs<br>. . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>22|
|4.3|Testin|g Details. . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>22|
||4.3.1|Unit Testing . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>22|
||4.3.2|Integration Testing<br>. . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
||4.3.3|User Interface Testing. . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
||4.3.4|Testing Checklist Summary . . . . . . . . . . . . . . .|. . . . . . . . .<br>23|
|**5**<br>**Con**|**clusion**|**s and Future Work**|**25**|
|5.1|Concl|usion<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>25|
|5.2|Future|Work . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . . . .<br>25|
|**Bibliog**|**raphy**||**26**|



ii 

# **List of Figures** 

|2.1|Use Case Diagram – StayBuddy<br>. . . . . . . . . . . . . . . . . . . . . . . . .|6|
|---|---|---|
|3.1|System Architecture Diagram – StayBuddy<br>. . . . . . . . . . . . . . . . . . .|10|
|3.2|Box and Line Diagram – StayBuddy . . . . . . . . . . . . . . . . . . . . . . .|10|
|3.3|Entity-Relationship (ER) Diagram – StayBuddy Database . . . . . . . . . . . .|11|
|3.4|Domain Model – StayBuddy . . . . . . . . . . . . . . . . . . . . . . . . . . .|12|
|3.5|Activity Diagram (Part 1) – Student Login, Hostel Search and Room Selection .|13|
|3.6|Activity Diagram (Part 2) – Room Preferences, Booking Confrmation and||
||Notifcations . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|14|
|3.7|Class Diagram – StayBuddy<br>. . . . . . . . . . . . . . . . . . . . . . . . . . .|15|
|3.8|Sequence Diagram – StayBuddy . . . . . . . . . . . . . . . . . . . . . . . . .|16|
|3.9|State Transition Diagram – StayBuddy . . . . . . . . . . . . . . . . . . . . . .|17|
|3.10|Context Diagram (Level 0 DFD) – StayBuddy . . . . . . . . . . . . . . . . . .|17|
|3.11|Level 1 Data Flow Diagram – StayBuddy<br>. . . . . . . . . . . . . . . . . . . .|18|
|3.12|Level 1 DFD (Detailed View) – StayBuddy<br>. . . . . . . . . . . . . . . . . . .|18|
|3.13|Level 2 Data Flow Diagram – StayBuddy<br>. . . . . . . . . . . . . . . . . . . .|18|
|3.14|Process 2 DFD – Recommendation and Booking Subsystem<br>. . . . . . . . . .|19|
|4.1|Intent Classifcation Example – DistilBERT Chatbot . . . . . . . . . . . . . . .|21|



iii 

# **List of Tables** 

|1.1|Comparison of Existing Solutions<br>. . . . . . . . . . . . . . . .|. . . . . . . .<br>2|
|---|---|---|
|1.2|Team Roles and Responsibilities . . . . . . . . . . . . . . . . .|. . . . . . . .<br>5|
|2.1|Event Response Table for Chatbot Intent Handling. . . . . . . .|. . . . . . . .<br>7|
|4.1|External APIs and SDKs Used in StayBuddy. . . . . . . . . . .|. . . . . . . .<br>22|



iv 

# **Chapter 1** 

# **Introduction** 

Finding suitable accommodation is a common challenge for many users, especially those who relocate to a new city for education or work. Users often rely on informal sources such as social media groups, online advertisements, or word of mouth to locate hostels. These methods frequently provide incomplete or unreliable information regarding hostel facilities, safety, cleanliness, and availability. 

StayBuddy is proposed as a smart hostel discovery and recommendation platform designed to help users easily find suitable accommodation. The system provides verified hostel listings, user reviews, booking functionality, and personalised recommendations based on user interactions. The main goal of StayBuddy is to simplify the hostel search process and provide users with reliable and data-driven suggestions when selecting accommodation. 

## **1.1 Background** 

The rapid growth of higher education in Pakistan has led to a surge in student migration to major cities like Islamabad, Lahore, and Karachi. According to the Higher Education Commission (HEC), over 1.6 million students are enrolled in universities across the country, with a significant percentage requiring off-campus accommodation. Existing solutions such as OLX, Zameen.com, and Facebook groups are fragmented, lack verification, and offer no intelligence. This gap motivated the development of StayBuddy – a centralised, AI-powered platform that streamlines hostel discovery, booking, and management. 

## **1.2 Existing Solutions** 

Several accommodation platforms exist such as Airbnb and Booking.com that allow users to search and book accommodation. These platforms provide detailed listings, ratings, and reviews from previous users. However, they mainly focus on short-term stays and tourism rather than long-term hostel accommodation. In many regions, hostel searching still depends on unstructured platforms such as Facebook groups, classified advertisements, or personal referrals. Research in recommendation systems shows that analysing user behaviour such as views, ratings, and bookings can help recommend suitable options. Collaborative filtering and hybrid 

1 

_StayBuddy – Smart Hostel Recommendation System_ 

_CHAPTER 1. INTRODUCTION_ 

recommendation models are widely used in modern recommendation platforms. StayBuddy aims to apply similar techniques to hostel discovery by analysing user interactions and reviews to generate intelligent hostel recommendations. 

**Table 1.1:** _Comparison of Existing Solutions_ 

|**System Name**|**System Overview**|**System Limitations**|
|---|---|---|
|Airbnb /|Global short-term rental|Focus on tourism; not optimised for student|
|Booking.com|platform with reviews and|hostels or long-term stays; no parent/warden|
||maps.|features.|
|OLX /|Classifed ads for property|Unverifed listings; no recommendation en-|
|Zameen.com|rental.|gine; no booking or complaint management.|
|Facebook Groups|Informal community pages<br>where hosts post<br>vacancies.|Unstructured; high spam; no trust or personal-<br>isation; no AI support.|



## **1.3 Problem Statement** 

Users currently face several difficulties when searching for hostels: 

- Lack of a centralised platform containing verified hostel listings. 

- Limited access to trustworthy reviews about hostel facilities. 

- Difficulty comparing hostels based on price, cleanliness, and facilities. 

- Lack of personalised recommendations based on user preferences. 

- Parents have no visibility into their child’s accommodation status. 

- Wardens lack efficient tools to manage complaints and attendance. 

Because of these limitations, users often choose accommodation that does not match their needs or expectations. The proposed StayBuddy system addresses these issues by providing a centralised platform where users can search hostels, read reviews, compare facilities, receive personalised recommendations, and manage bookings with full stakeholder involvement (students, parents, wardens, owners). 

2 

_StayBuddy – Smart Hostel Recommendation System_ 

_CHAPTER 1. INTRODUCTION_ 

## **1.4 Scope** 

The scope of the StayBuddy system includes: 

- Hostel discovery and listing management. 

- User interaction tracking. 

- Hostel reviews and ratings. 

- Booking management with AI-assisted room selection. 

- Complaint management with AI-suggested resolutions. 

- Recommendation system for personalised hostel suggestions. 

- Parent portal for monitoring child’s attendance and bookings. 

- Warden dashboard for managing complaints, attendance, and announcements. 

- Owner module for adding and managing hostels. 

- Admin portal for verification and user management. 

The system is developed as a web-based application consisting of a React frontend, FastAPI backend, PostgreSQL database, and Python-based AI microservices for recommendations and natural language processing. 

## **1.5 Modules** 

### **1.5.1 User Management Module** 

Handles user registration, login, profile management, and role-based access control (Student, Parent, Warden, Owner, Admin). 

- JWT-based authentication with role-specific dashboards. 

- Student preference management (budget, amenities, commute distance). 

### **1.5.2 Hostel Management Module** 

Stores hostel information including location, capacity, facilities, room types, pricing, and verification status. 

- Owner adds/edits hostel details. 

3 

_StayBuddy – Smart Hostel Recommendation System_ 

_CHAPTER 1. INTRODUCTION_ 

- Admin verifies hostels before public listing. 

### **1.5.3 Booking Management Module** 

Allows users to reserve rooms in hostels, select room types, and request room changes. 

- AI-assisted room selection based on preferences (quiet floor, window view, etc.). 

- Booking confirmation and cancellation. 

### **1.5.4 Review and Rating Module** 

Allows users to submit reviews and ratings for hostels after staying. 

- Multi-dimensional ratings (cleanliness, facilities, management, location). 

- Text reviews with sentiment analysis (planned for future). 

### **1.5.5 Complaint Management Module** 

Enables users to submit complaints regarding hostel services or facilities, with AI-generated suggested actions for wardens. 

- Automatic complaint categorisation using NLP. 

- Status tracking (Pending, In Progress, Resolved). 

### **1.5.6 Recommendation Module** 

Analyses user interactions and preferences to generate personalised hostel recommendations using a hybrid model (content-based + collaborative filtering). 

- Match score (0–1) and explainable factors (e.g., “within budget”, “has WiFi”). 

- Cold-start handling using content-based only. 

### **1.5.7 Communication and Notification Module** 

Handles real-time announcements from wardens, chatbot interactions, and push notifications for parents and wardens. 

- AI chatbot for natural language queries. 

- Emergency SOS alerts to warden and parent. 

4 

_StayBuddy – Smart Hostel Recommendation System_ 

_CHAPTER 1. INTRODUCTION_ 

## **1.6 Work Division** 

**Table 1.2:** _Team Roles and Responsibilities_ 

|**Team Member**|**Responsibilities**|
|---|---|
|Eraj Zaman (221-1296)|Recommendation engine API, Student mobile/web app (all screens),<br>Voice search (optional), AI model integration.|
|Samiya Saleem|Chatbot completion (entity extraction, response templates, API), Chat-|
|(221-1065)|bot UI, Parent portal (all screens), Complaint submission fow, UI/UX<br>consistency.|
|Zarnab (221-1005)|All backend APIs (real DB responses, authentication), Warden dash-<br>board, Admin portal, Database integrity, Deployment.|



5 



<!-- Start of picture text -->
student |_ Admin Parent Warden<br>| t + + StayBuddy System1 SS— [ | :—— me<br>File Complain via Chatbot Book Hostelsetwith Al Room Search& Filter Hostels ViewAl Recommendations Verity Hostets View Child Status& Alerts + Register/Login ls Post Announcements Manage Complaints<br><!-- End of picture text -->

_StayBuddy – Smart Hostel Recommendation System_ 

_CHAPTER 2. PROJECT REQUIREMENTS_ 

### **2.1.2 Event Response Table** 

**Table 2.1:** _Event Response Table for Chatbot Intent Handling_ 

|**Event**|**Event (User**|**Condition**|**System Response**|
|---|---|---|---|
|**ID**|**Message)**|||
|E01|“Show me hostels|Intent =|Extract budget=15000, location=FAST.|
||under 15000 near<br>FAST”|hostel_search|Query DB and return list.|
|E02|“Does Green Valley<br>have a gym?”|Intent =<br>amenity_inquiry|Extract hostel, amenity; respond yes/no.|
|E03|“How much is a single<br>room at Sunrise?”|Intent = pric-<br>ing_information|Fetch price and respond.|
|E04|“The WiFi is not<br>working”|Intent =<br>complaint_issue|Auto-create complaint, return ID.|
|E05|(Unknown message)|Intent = unknown|Fallback response: list capabilities.|



## **2.2 Functional Requirements** 

### **2.2.1 User Management Module** 

**FR1** The system shall allow users to register and login with role-based access (Student, Parent, Warden, Owner, Admin). 

**FR2** The system shall allow students to set and update their preferences (budget, amenities, commute distance). 

### **2.2.2 Hostel Management Module** 

**FR3** The system shall display a list of available hostels with filtering and sorting. **FR4** The system shall allow owners to add and edit hostel information. **FR5** The system shall allow admins to verify hostels. 

7 

_StayBuddy – Smart Hostel Recommendation System_ 

_CHAPTER 2. PROJECT REQUIREMENTS_ 

### **2.2.3 Recommendation Module** 

- **FR6** The system shall record user interactions such as viewing, favouriting, and booking hostels. 

- **FR7** The system shall generate personalised hostel recommendations using a hybrid model (content-based + collaborative filtering) with match scores and explainable factors. 

### **2.2.4 Booking Module** 

- **FR8** The system shall allow students to book hostel rooms and select room types. 

- **FR9** The system shall provide AI-assisted room selection based on student preferences (quiet floor, near window, etc.). 

- **FR10** The system shall allow students to request room changes. 

### **2.2.5 Complaint and Chatbot Module** 

- **FR11** The system shall allow students to submit complaints via the chatbot or form. 

- **FR12** The system shall automatically categorise complaints and generate AI-suggested resolutions for wardens. 

- **FR13** The system shall provide a natural language chatbot capable of answering queries about hostels, amenities, pricing, and bookings. 

### **2.2.6 Warden and Parent Modules** 

- **FR14** The system shall allow wardens to view and manage complaints, mark attendance, post announcements, and allocate rooms. 

- **FR15** The system shall allow parents to view their child’s attendance, booking status, and receive emergency alerts. 

## **2.3 Non-Functional Requirements** 

### **2.3.1 Usability** 

- **USE1** The system shall provide a user-friendly interface with responsive design for mobile and desktop. 

- **USE2** The chatbot shall respond to user queries in under 3 seconds and offer quick reply buttons. 

8 

_StayBuddy – Smart Hostel Recommendation System_ 

_CHAPTER 2. PROJECT REQUIREMENTS_ 

### **2.3.2 Performance** 

**PER1** 95% of API requests shall complete within 2 seconds over a 20 Mbps connection. **PER2** The recommendation engine shall return top-10 results within 1 second for 100 concurrent users. 

### **2.3.3 Security** 

**SEC1** All passwords shall be hashed using bcrypt before storage. 

**SEC2** API endpoints shall require JWT tokens and enforce role-based access control. 

### **2.3.4 Availability** 

**AVAIL1** The system shall have 99.9% uptime during demonstration hours. 

> **AVAIL2** Critical features (login, search, booking) shall remain functional even if the recommendation service is temporarily down (fallback to content-based only). 

9 



<!-- Start of picture text -->
Presentation Layer<br>(React)\nWeb/MobileStudent App Warden Dashboard (React) Parent Portal (React) Admin. Portal (React)<br>SS<br>ML — Authentication Service<br> Service (Recs + Chatbot) Complaint Service Booking Service Hostel Management (wT)<br>OS[—EEEE——=———————_—PostgreSQL Database<br><!-- End of picture text -->



<!-- Start of picture text -->
PostgreSQL<br>Frontend\n(React Web App) REST API\n(FastAPI( . /<br>Node. js)<br>ML<br>Microservice\n(Recommendz External NLP<br>+ Chatbot) Models\n(DistilBERT)<br><!-- End of picture text -->



<!-- Start of picture text -->
HOSTEL<br>USER int |T id Pk<br>string | name string |4 name<br>string | | string | address<br>string || emaitrole | UK student, parent, warden, admin int |4  wardenid FK<br>l it string | verification_status<br>extends files (as student) receives ‘contains<br>0SS aA 9 »<br>COMPLAINT 2<br>int | user_idSTUDENT PK,FK intint || idstudentid | PkFK : int NOTIFICATIONid Pk f int | id ROOM_TYPE4 Pk<br>f t { ] t t int | hostel_id Fk<br>can be parent of string | university int | hostelid Fk int user_id | Fk places (as student) f t<br>int | budget_max string | category Internet, Cleantiness, etc. string | message string | type_name Single, Double, Dorm<br>f . | _t int | price_per_month<br>text | preference_amenities text | ai_suggestion boolean | is_read f<br>string | status pending, in_progress, resolved<br>j | int | available_beds<br>IsChild of 3booked in<br>BOOKING<br>intPARENT_STUDENT_LINK| id PK | int | id +—+Pk<br>int | parentid |—FK int | student_id | FK<br>int |+ student_id |—Fk int |, room_type_id |{1FK<br>string | status pending, confirmed, cancelled<br><!-- End of picture text -->



<!-- Start of picture text -->
aePARENT [me<br>EES<br>monitors<br>O<br>AM HOSTEL<br>STUDENT coe<br>receives<br>maké<br>fe files generates h= anagea_byd_b'<br>ae? O 0 7 O<br>BOOKING N COMPLAINT<br>WARDEN<br><!-- End of picture text -->



<!-- Start of picture text -->
Start<br>|<br>*<br>Student logs into StayBuddy<br>( ie<br>|j b<br>Valid credentials?<br>/ \<br>yt Yes<br>nT %<br>a Display personalized home<br>Show error message screen with Al<br>recommendations<br>wen—™. * ~<br>/Ss | / ca‘b<br>||<br>Student action<br>Searches/filters \<br>$ |<br>Apply filters: price,<br>distance, amenities<br>!<br>No Selects a recommended<br>hostel<br>Display filtered hostel list |<br>| }<br>ec ff<br>View hostel detail page<br>a ||<br>ce ’<br>Decides to book?<br>|<br>Yes<br>Select room type:<br>Single /Double/Dorm<br>”<br>|<br>’<br>Has room preferences?<br>Fr \\<br>7 \ \<br><!-- End of picture text -->



<!-- Start of picture text -->
y |<br>Has room preferences?<br>“a Al-Assisted Room<br>Student manually picks Selection\nSystem suggests<br>available bed best bed based on\nquiet<br>L = floor, window view, etc.<br>Student confirms room<br>choke<br>Enter check-in date and<br>duration<br>System calculates total<br>price<br>Show booking surnmary<br>Confirm booking?<br>Yes<br>Create booking record in<br>database<br>Send notifications:\n-<br>Warden dashboard\n-<br>Parent portal<br>Display booking<br>confirmation screen\nwith<br>Booking ID<br>End<br><!-- End of picture text -->



<!-- Start of picture text -->
+int id<br>+string name<br>+string email<br>+string role<br>Qe<br>+login()<br>+logout()<br>/\<br>+int linkedStudentld<br>+viewChildStatus()<br>+receiveAlerts()<br>+string university +int id<br>+int budgetMax +string name<br>+float commuteMaxKm +string address<br>+list amenitiesPref +float distanceFromCampus<br>+float overallRating<br>+viewRecommendations() +list amenities<br>+bookHostel() +list roomTypes<br>+fileComplaint()<br>contains<br>makes1 ' \ receives managed by<br>0..* 0. files 0."<br>0 1<br>+int id +int id<br>+date checkInDate +string category<br>+int durationMonths +string description<br>+string status +string aiSuggestion +manageComplaints()<br>+string status +allocateRoom()<br>+confirm() +postAnnouncement()<br>+cancel() +updateStatus()<br><!-- End of picture text -->



<!-- Start of picture text -->
Student Student App (Frontend) Backend API (Zarnab) Recommendation Engine (Eraj) Database<br>—_-eareesewors>*r0Logs in & views Home screen ><br>___ GET /api/recommendations (with JWT token)<br>Verify JWT & extract student_id<br>_ EF POST /recommendations (student_id)<br>Fetch student preferences & interaction history<br>—__-errerreeesss'vw*wtrerrrrrm<br>Returns student profile & past views<br>Runs Hybrid Model (Content-Based + Collaborative Filtering)<br>Returns ranked list of hostels with match scores & factors<br>Fetch full details for each recommended hostel<br>eeeeee el SS=»<br>Returns hostel details (name, price, amenities, etc.)<br>Returns personalized recommendations with match scores<br>Displays "Recommended for You" list<br>oe<br>Student Student App (Frontend) Backend API (Zarnab) Recommendation Engine (Eraj) Database<br><!-- End of picture text -->



<!-- Start of picture text -->
Student files complaint<br>Pending<br>ae veo = Warden acknowledges and<br>Pag Warden reopens complaint starts work<br>,<br>fi<br>1<br>'<br>'<br>Al suggestion88 =<br>enerated\nand ; Warden resolves directly<br>displayed In_Progress (rare)rare<br>to warden :<br>i<br>, “<br>,<br>,<br>’<br>' Warden marks as resolved<br>i<br>1<br>1<br>'<br>i<br>Warden notes<br>added\nParent/student Resolved<br>notified S<br>,<br>\<br>Complaint. closed \f1<br>i<br>i<br>f<br>1<br>| Resolution comment<br>added\nNotification sent<br><!-- End of picture text -->



<!-- Start of picture text -->
External Entities<br>(student<br>Login Credentials, Search Recommendations, Hostel<br>Queries, Preferences, Details, Booking<br>Booking Requests, Confirmation, Chatbot<br>Complaints Responses<br>‘StayBuddy System (Process<br>0.0 Hostel<br>Recommendation &<br>Management System<br>SSS=— a iha aaa Verlication Status, Basic gj, Verification Queue, User HACE<br>Parent. Warden Hostel Owner ‘System Admin.<br><!-- End of picture text -->



<!-- Start of picture text -->
Alerts, —<br>Requests, Preferences Vicw- Stats<br>StayBuddy<br>Recommendations,<br>Confirmation Complaints, Analytics<br>Manage Hostel<br><!-- End of picture text -->



<!-- Start of picture text -->
L— — —=——=S — —ee - ; a le spin<br><!-- End of picture text -->



<!-- Start of picture text -->
1.0 Authentication & ,—| 2.0 Recommendation =— 3.0 Booking & Room SS———] eneee<br>5.0 Notificationlll Service |__| yo pataI ste ata PAA | Complaint Data| A! SuBgestion Update Status<br>ONi i<br>—_~6C~CSCS~S<S S]=p<br><!-- End of picture text -->



<!-- Start of picture text -->
Process 2.0 (Expanded)<br>2.1\nReceive Search Filters<br>/ Student ID<br>2.2\nFetch Student Prefs<br>&\nInteraction History<br>( 2.3\nRun Hybrid<br>Read Read Model\n(Content +<br>Collaborative)<br>Read Student ID or Filters<br>DataStores<br>a, —_ | <<: ae > 2.4\nGenerate Match<br>D1\nUser Profiles D5\nInteraction History D2\nHostel Listings Saris ENISHI<br>a a = ? QO ? Factors<br>2.5\nFilter & Rank Hostels |<br>Ranked Hostels + Match<br>Scores<br>Student<br><!-- End of picture text -->

# **Chapter 4** 

# **Implementation and Testing** 

This chapter describes the algorithms, external APIs, and testing procedures used in StayBuddy. 

## **4.1 Algorithm Design** 

### **4.1.1 Hybrid Recommendation Algorithm** 

The recommendation engine combines content-based filtering and collaborative filtering using a weighted hybrid approach. Content-based filtering computes similarity between a student’s preference profile and hostel feature vectors using cosine similarity. Collaborative filtering leverages patterns from other users with similar behaviour using k-Nearest Neighbours (k-NN). The final score is a weighted combination: 



where _α_ is dynamically adjusted based on the number of user interactions (cold-start uses _α_ = 1 _._ 0). 

### **4.1.2 Intent Classification for Chatbot** 

The chatbot uses a fine-tuned DistilBERT model for intent classification. Figure 4.1 illustrates an example of the algorithm’s classification output. 

20 

=| Algorithm Hybrid Recommendatio ¢& ar File Edit View Hy =» BIS ® By & Algorithm: Hybrid Recommendation for StayBuddy Input: student_id, top_k = 10 Output: Ranked list of hostels with match scores and top matching factors 1. Load student preferences from database (budget, distance, amenities, study/social score) 2. Load student interaction history (views, favorites, bookings) 3. Load hostel feature matrix (price, distance, amenities, ratings) // Content-Based Filtering For each hostel in all_hostels: content_score = cosine_similarity(student_preference_vector, hostel_feature_vector) // Collaborative Filtering if student has interaction history: similar_students = find_k_nearest_neighbors(student, interaction_matrix) collab_score = weighted_average_rating(similar_students, hostel) else: collab_score = 9.5 // neutral for new users (cold start) // Hybrid Score hybrid_score = a * content_score + (1-a) * collab_score // a is weight (e.g., 0.6) // Generate explainable factors matching factors = [] if hostel.price <= student.budget_max: matching_factors.append("“within budget") if hostel.wifi and student.prefers_wifi: matching_factors.append("“has WiFi") if hostel.distance <= student.commute_max_km: matching_factors.append("near campus") // Sort by hybrid_score descending and return top_k return sorted_hostels[@:top_k] with scores and matching factors 



_StayBuddy – Smart Hostel Recommendation SystemCHAPTER 4. IMPLEMENTATION AND TESTING_ 

## **4.2 External APIs and SDKs** 

**Table 4.1:** _External APIs and SDKs Used in StayBuddy_ 

|**API / SDK**|**Description**|**Purpose**|**Endpoint / Function**|
|---|---|---|---|
|**(version)**||||
|FastAPI (0.95+)|Python web<br>framework|Backend API<br>development|FastAPI(), router, dependency<br>injection|
|JWT (PyJWT|JSON Web Token|Authentication &|jwt.encode(),jwt.decode()|
|2.6+)|library|authorisation||
|bcrypt (4.0+)|Password hashing<br>library|Secure password<br>storage|bcrypt.hashpw(),checkpw()|
|DistilBERT|Pre-trained|Intent|pipeline("text-classification")|
|(HuggingFace)|transformer|classifcation||
|scikit-learn (1.2+)|ML library|Recommendation<br>engine|cosine_similarity(),<br>NearestNeighbors|
|PostgreSQL|Relational DB|Database|psycopg2.connect(),|
|(psycopg2)|driver|connection|cursor.execute()|
|React (18+)|JavaScript<br>frontend|Student app,<br>dashboards|useState,useEffect, Axios|
|Axios|HTTP client|API calls from<br>frontend|axios.get(),axios.post()|



## **4.3 Testing Details** 

### **4.3.1 Unit Testing** 

Unit tests were written using PyTest for backend APIs and Jest for frontend components. Below is an example test for the recommendation API. 

~~<mark>�</mark>~~ <mark>�</mark> 

1 # test_recommendation_api .py 2 import pytest 

3 from fastapi.testclient import TestClient 4 from app.main import app 

22 

_StayBuddy – Smart Hostel Recommendation SystemCHAPTER 4. IMPLEMENTATION AND TESTING_ 

5 6 client = TestClient(app) 7 8 def test_get_recommendations_valid_student (): 9 response = client.post( "/api/recommendations" , 10 json ={ "student_id" : 123, "top_k" : 10}) 11 assert response.status_code == 200 12 data = response.json () 13 assert "recommendations" in data 14 assert len (data[ "recommendations" ]) <= 10 15 first = data[ "recommendations" ][0] 16 assert "match_score" in first 17 assert 0 <= first[ "match_score" ] <= 1 18 assert " top_matching_factors" in first 19 assert isinstance (first[ "top_matching_factors" ], list ) 20 21 def test_get_recommendations_invalid_student (): 22 response = client.post( "/api/recommendations" , 23 json ={ "student_id" : 99999 , "top_k" : 5}) 24 assert response.status_code == 404 25 assert "error" in response.json () ~~<mark>�</mark>~~ <mark>�</mark> 

Listing 4.2: Unit Test for Recommendation API (PyTest) 

### **4.3.2 Integration Testing** 

Integration tests verified that frontend components correctly communicate with backend APIs and that the recommendation microservice returns expected data. All endpoints were tested using Postman collections. 

### **4.3.3 User Interface Testing** 

The UI was tested across multiple browsers (Chrome, Firefox) and device sizes (mobile, tablet, desktop). All critical user flows were verified end-to-end. 

### **4.3.4 Testing Checklist Summary** 

- ✓ Authentication: Login/register for all roles. 

- ✓ Recommendation engine returns ranked hostels with match scores. 

- ✓ Chatbot responds correctly to 7+ intents. 

23 

_StayBuddy – Smart Hostel Recommendation SystemCHAPTER 4. IMPLEMENTATION AND TESTING_ 

- ✓ Booking flow completes and stores in DB. 

- ✓ Complaint filing creates record and AI suggestion. 

- ✓ Warden can update complaint status. 

- ✓ Parent receives notifications. 

- ✓ Emergency SOS triggers alerts. 

- ✓ All API endpoints return proper HTTP status codes. 

24 

# **Chapter 5** 

# **Conclusions and Future Work** 

## **5.1 Conclusion** 

StayBuddy has successfully evolved from a concept into a functionally rich, AI-integrated platform that addresses the critical problem of finding suitable student accommodation. As demonstrated in the mid-term and final implementation, the project has delivered a fully connected ecosystem where: 

- Students receive intelligent, personalised hostel recommendations with explainable match scores and AI-assisted room selection. 

- An NLP-powered chatbot understands natural language queries, answers questions, and automatically files complaints. 

- Wardens are equipped with a powerful dashboard to manage complaints (with AI-suggested resolutions), allocate rooms, post announcements, and track attendance. 

- Parents can monitor their child’s attendance, bookings, and receive real-time emergency alerts. 

- Owners can add and manage hostel listings, with admin verification ensuring trust. 

The system’s three-tier architecture, JWT-based authentication, and role-based access control provide a secure and scalable foundation. The hybrid recommendation model (content-based + collaborative filtering) achieves high precision and coverage, while the DistilBERT-based chatbot demonstrates robust intent classification accuracy. All components have been integrated and tested through a complete student journey from registration to booking confirmation. 

StayBuddy is not just a listing platform; it is an intelligent, data-driven solution that brings transparency, efficiency, and trust to the student housing market, benefiting all stakeholders. 

## **5.2 Future Work** 

Building upon the current implementation, the following enhancements are planned: 

25 

_StayBuddy – Smart Hostel Recommendation SystemCHAPTER 5. CONCLUSIONS AND FUTURE WORK_ 

- **Owner Module Completion:** Full management dashboard for owners to edit hostels, manage rooms, and view analytics (occupancy, revenue). 

- **Real-time Push Notifications:** Integration with Firebase Cloud Messaging (FCM) for booking confirmations, complaint updates, and emergency alerts. 

- **Voice Search:** Allow students to search for hostels and interact with the chatbot using voice commands. 

- **Social and Contextual Recommendations:** Incorporate friend networks, semester schedules, and weather data into recommendations. 

- **Biometric Attendance:** Integration with RFID or face recognition for automated checkin/check-out. 

- **Mobile Native Apps:** Develop iOS and Android apps using React Native for better performance and offline support. 

- **Advanced Analytics Dashboard:** For owners and admins, including predictive occupancy models and revenue forecasting. 

26 

# **Bibliography** 

- [1] Higher Education Commission of Pakistan (HEC). _Annual Report 2022–2023: University Enrolment Statistics_ . Islamabad: HEC, 2023. 

- [2] F. Ricci, L. Rokach, and B. Shapira. _Recommender Systems Handbook_ , 2nd ed. New York: Springer, 2015. 

- [3] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova. “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.” In _Proceedings of NAACL-HLT 2019_ , pp. 4171–4186, 2019. 

- [4] S. Ramirez. _FastAPI Documentation_ . https://fastapi.tiangolo.com , 2024. 

27 

