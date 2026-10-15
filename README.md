# 🎓 AlphabetAI Attendance — AI-Powered Smart Attendance System

> A complete, responsive, full-stack hackathon web application built with **Python Flask**, **MongoDB Community**, **OpenCV Deep Learning (YuNet + SFace)**, and **vanilla HTML5 / CSS3 / JavaScript**.

---

## 🌟 Key Features

1. **Teacher Login & Secure Authentication**:
   - Institutional faculty login with hashed passwords (`werkzeug.security.generate_password_hash`).
   - Pre-seeded default faculty credentials (`teacher@alphabet.edu` / `Admin@123`) plus on-page faculty registration.
   - Session-protected endpoints with `@login_required` decorators.

2. **Student Directory & Duplicate Prevention**:
   - Unique constraints on **Roll Number** and **Email** preventing student duplication.
   - Real-time search by student name, roll number, or department.
   - Student profile editing and permanent deletion with confirmation.

3. **Automatic A–Z Alphabetical Sorting**:
   - **All views** (Live Attendance Queue, Student Directory, Cumulative Reports, and CSV exports) are strictly sorted **alphabetically from A to Z** by student name.
   - Interactive Alphabet Jump Bar (`All`, `A`, `B`, `C` ... `Z`) for instant letter-based filtering.

4. **Genuine Local AI Face Recognition (No Paid APIs)**:
   - **OpenCV Deep Learning (DNN)** using official OpenCV Zoo models:
     - **YuNet ONNX**: High-speed neural face detection and 5-point facial landmark alignment.
     - **SFace ONNX**: Deep metric learning model extracting genuine **128-dimensional L2-normalized biometric embeddings**.
   - **True Cosine Similarity Matching**: Matches faces against stored 128-d vectors using exact metric thresholding. No fake checks or mock detection.
   - 100% free and locally runnable; no external or paid cloud APIs needed.

5. **Live Alphabetical Attendance Queue**:
   - Displays all registered students in real-time with status badges:
     - `✅ Present` (Emerald Green)
     - `❌ Absent` (Crimson Red)
     - `⏳ Pending` (Amber)
   - Real-time quick-action buttons for marking Present or Absent, plus manual correction dialog with reason logging.

6. **Duplicate Attendance Prevention & Audit Logging**:
   - **Strict duplicate guard**: Once a student is marked `Present` on the current date, any subsequent scan or mark attempt is blocked and alerted with a timestamp warning.
   - **Comprehensive Security Audit Trail**: Logs every action (`AUTH_LOGIN`, `STUDENT_ENROLLED`, `FACE_VERIFIED_ATTENDANCE`, `DUPLICATE_ATTENDANCE_BLOCKED`, `MANUAL_CORRECTION`) with timestamp, teacher ID, student details, AI confidence %, and IP address.

7. **Dashboard with Real Database Statistics**:
   - Real-time KPI summary cards: Total registered students, Today's Present count, Today's Absent count, Attendance Rate (%).
   - Live system status pill showing MongoDB connection status, AI model status, and camera resolution.

8. **Attendance History, Percentages & CSV Export**:
   - Historical date picker to inspect past session records.
   - Cumulative student attendance percentage tracker (e.g. `90.0%`) with automatic eligibility badges (`Eligible >= 75%` vs `Shortage < 75%`).
   - One-click **Daily Attendance CSV** and **Cumulative Summary CSV** downloads.

9. **Responsive Blue-and-White Theme**:
   - Modern academic design with deep navy, royal blue, sky blue, and crisp white cards.
   - Fully responsive for mobile devices, tablets, and desktop displays.

10. **Camera Permission Handling & Manual Fallback**:
    - Gracefully handles camera permission denials.
    - Built-in **Photo Upload fallback** allows testing AI recognition via uploaded image files if a live camera is blocked or unavailable.
    - Direct **Manual Attendance fallback** allows teachers to mark any student directly with audit notation.

11. **Biometric Data Protection & Explicit Consent**:
    - Enrollment requires mandatory confirmation of student consent for extracting mathematically irreversible 128-d feature embeddings.
    - Consent timestamp recorded in database.

---

## 💻 Windows 11 Simple Setup & Run Guide

### 1. Prerequisites
- **Windows 11** with **Python 3.10+** (Python 3.12 verified).

### 2. Open PowerShell in Project Directory
Open PowerShell or Windows Terminal in the project folder:
```powershell
cd c:\Users\OMKAR.MUDDADA\OneDrive\Desktop\aiteac
```

### 3. Install Dependencies
Install all required libraries using `pip`:
```powershell
pip install -r requirements.txt
```
*(Packages: Flask, pymongo, mongomock, opencv-python, numpy, pillow, werkzeug)*

### 4. MongoDB Configuration

#### Option A: Zero-Config In-Memory Mode (Default Fallback)
If you do not have MongoDB Community Server installed or running, **AlphabetAI Attendance automatically runs seamlessly in In-Memory Mode using `mongomock`**. No installation or configuration is required!

#### Option B: Live MongoDB Community Server (Recommended for Persistent Production)
1. Download and install [MongoDB Community Server for Windows](https://www.mongodb.com/try/download/community).
2. Start the MongoDB service from PowerShell:
   ```powershell
   net start MongoDB
   ```
   Or run `mongod` manually:
   ```powershell
   & "C:\Program Files\MongoDB\Server\7.0\bin\mongod.exe" --dbpath "C:\data\db"
   ```
3. Set your custom connection string (optional, defaults to `mongodb://127.0.0.1:27017/alphabetai_db`):
   ```powershell
   $env:MONGO_URI = "mongodb://127.0.0.1:27017/alphabetai_db"
   ```

### 5. Run the Application
Launch the Flask server:
```powershell
python app.py
```

Console output will display:
```text
======================================================================
  AlphabetAI Attendance — AI-Powered Smart Attendance System
  Database Engine: MongoDB Community (Local 27017)  [or In-Memory Mock]
  AI Facial Model: OpenCV YuNet + SFace (Genuine Local AI)
  Access Portal:   http://127.0.0.1:5000
  Faculty Access:  teacher@alphabet.edu / Admin@123
======================================================================
```

Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

### 6. Log In
- **Email**: `teacher@alphabet.edu`
- **Password**: `Admin@123`

---

## 🧪 Automated Testing

To run the complete automated test suite verifying authentication, duplicate prevention, A-Z sorting, manual overrides, and CSV reports:

```powershell
python test_app.py
```

---

## 📁 Project Architecture

```text
aiteac/
│
├── app.py                      # Complete single-file application (Backend + UI + AI)
├── requirements.txt            # Python dependencies
├── test_app.py                 # Automated unit and integration test suite
├── README.md                   # Setup and evaluation documentation
└── models/                     # Official local OpenCV Zoo neural network weights
    ├── face_detection_yunet_2023mar.onnx     (YuNet Face Detector & Landmark model)
    └── face_recognition_sface_2021dec.onnx   (SFace 128-d Biometric Recognizer)
```
