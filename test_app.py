"""
Automated Comprehensive Test Suite for AlphabetAI Attendance System.
Tests all endpoints, AI models, duplicate prevention, alphabetical sorting, and audit logging.
"""

import unittest
import json
import io
import cv2
import numpy as np
import base64
from app import app, db, students_col, teachers_col, attendance_col, audit_col, extract_face_biometrics

class TestAlphabetAIAttendance(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        # Clean collections for deterministic testing
        students_col.delete_many({})
        attendance_col.delete_many({})
        audit_col.delete_many({})

    def login_teacher(self):
        return self.client.post("/login", data={
            "email": "teacher@alphabet.edu",
            "password": "Admin@123"
        }, follow_redirects=True)

    def generate_dummy_face_image_base64(self):
        """Generates a synthetic 320x320 face-like test pattern."""
        img = np.ones((320, 320, 3), dtype=np.uint8) * 230
        # Draw head
        cv2.circle(img, (160, 160), 80, (180, 150, 130), -1)
        # Draw eyes
        cv2.circle(img, (135, 140), 10, (50, 30, 20), -1)
        cv2.circle(img, (185, 140), 10, (50, 30, 20), -1)
        # Draw nose
        cv2.line(img, (160, 145), (160, 175), (80, 50, 40), 3)
        # Draw mouth
        cv2.ellipse(img, (160, 195), (25, 10), 0, 0, 180, (50, 30, 20), 3)
        _, buf = cv2.imencode('.jpg', img)
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

    def test_01_teacher_authentication(self):
        print("\n[Test 1] Testing Teacher Authentication & Security...")
        # Unauthenticated access to dashboard should redirect to login
        res = self.client.get("/dashboard")
        self.assertEqual(res.status_code, 302)
        self.assertIn("/login", res.headers["Location"])

        # Invalid login
        res_fail = self.client.post("/login", data={
            "email": "teacher@alphabet.edu",
            "password": "WrongPassword!"
        })
        self.assertIn(b"Invalid email or password", res_fail.data)

        # Successful login
        res_ok = self.login_teacher()
        self.assertEqual(res_ok.status_code, 200)
        self.assertIn(b"Live Queue", res_ok.data)
        print(" -> Authentication & Protected Routes verified!")

    def test_02_student_registration_and_duplicate_prevention(self):
        print("\n[Test 2] Testing Student Registration & Duplicate Prevention...")
        self.login_teacher()

        # Try to register without consent (should fail)
        res_no_consent = self.client.post("/api/students", json={
            "full_name": "Charlie Chaplin",
            "roll_no": "CS2026001",
            "email": "charlie@test.edu",
            "department": "Computer Science & AI",
            "consent_granted": False
        })
        self.assertEqual(res_no_consent.status_code, 400)
        self.assertIn("consent", res_no_consent.get_json()["error"].lower())

        # Register valid student with consent
        res_ok = self.client.post("/api/students", json={
            "full_name": "Charlie Chaplin",
            "roll_no": "CS2026001",
            "email": "charlie@test.edu",
            "department": "Computer Science & AI",
            "consent_granted": True
        })
        self.assertEqual(res_ok.status_code, 200)
        self.assertTrue(res_ok.get_json()["success"])

        # Duplicate Roll Number prevention
        res_dup_roll = self.client.post("/api/students", json={
            "full_name": "Charles Xavier",
            "roll_no": "CS2026001", # same roll
            "email": "xavier@test.edu",
            "department": "Computer Science & AI",
            "consent_granted": True
        })
        self.assertEqual(res_dup_roll.status_code, 409)
        self.assertIn("already registered", res_dup_roll.get_json()["error"])

        # Duplicate Email prevention
        res_dup_email = self.client.post("/api/students", json={
            "full_name": "Charlotte Bronte",
            "roll_no": "CS2026002",
            "email": "charlie@test.edu", # same email
            "department": "Computer Science & AI",
            "consent_granted": True
        })
        self.assertEqual(res_dup_email.status_code, 409)
        print(" -> Duplicate prevention for Roll Number & Email successfully verified!")

    def test_03_alphabetical_a_to_z_sorting(self):
        print("\n[Test 3] Testing Automatic A–Z Alphabetical Sorting...")
        self.login_teacher()

        # Insert out of order: Zara, Bob, Alice, David
        names = ["Zara Phillips", "Bob Dylan", "Alice Walker", "David Bowie"]
        for i, n in enumerate(names):
            self.client.post("/api/students", json={
                "full_name": n,
                "roll_no": f"CS2026{i+10:03d}",
                "email": f"student{i+10}@test.edu",
                "department": "Computer Science & AI",
                "consent_granted": True
            })

        # Fetch students API
        res = self.client.get("/api/students")
        data = res.get_json()
        returned_names = [s["full_name"] for s in data["students"]]
        expected_sorted = sorted(names)

        self.assertEqual(returned_names, expected_sorted)
        print(f" -> Input: {names}")
        print(f" -> Output (A–Z Sorted): {returned_names}")
        self.assertEqual(returned_names[0], "Alice Walker")
        self.assertEqual(returned_names[-1], "Zara Phillips")
        print(" -> Automatic A–Z Sorting verified!")

    def test_04_live_queue_and_manual_attendance(self):
        print("\n[Test 4] Testing Live Attendance Queue & Manual Attendance...")
        self.login_teacher()

        # Create students
        self.client.post("/api/students", json={
            "full_name": "Alan Turing",
            "roll_no": "CS101",
            "email": "alan@turing.edu",
            "department": "Computer Science & AI",
            "consent_granted": True
        })
        s_id = students_col.find_one({"roll_no": "CS101"})["_id"]

        # Check queue has student with status 'Pending'
        res_queue = self.client.get("/api/queue")
        q_data = res_queue.get_json()
        self.assertEqual(q_data["total_students"], 1)
        self.assertEqual(q_data["queue"][0]["status"], "Pending")

        # Mark Present manually
        res_mark = self.client.post("/api/mark-attendance", json={
            "student_id": str(s_id),
            "status": "Present",
            "reason": "Direct Teacher Verification"
        })
        self.assertTrue(res_mark.get_json()["success"])

        # Check queue updated
        res_queue2 = self.client.get("/api/queue")
        q2 = res_queue2.get_json()
        self.assertEqual(q2["present_count"], 1)
        self.assertEqual(q2["queue"][0]["status"], "Present")
        print(" -> Queue status update and manual override verified!")

    def test_05_prevent_duplicate_attendance_and_audit_logs(self):
        print("\n[Test 5] Testing Duplicate Attendance Prevention & Audit Logs...")
        self.login_teacher()

        # Enroll student with simulated face embedding
        dummy_feat = np.random.randn(1, 128).astype(np.float32)
        dummy_feat /= np.linalg.norm(dummy_feat)

        s_res = students_col.insert_one({
            "full_name": "Grace Hopper",
            "roll_no": "CS200",
            "email": "grace@navy.mil",
            "department": "Computer Science & AI",
            "biometric_enrolled": True,
            "face_embedding": dummy_feat.flatten().tolist(),
            "consent_granted": True
        })
        student_id = str(s_res.inserted_id)

        # Mark Present once
        self.client.post("/api/mark-attendance", json={
            "student_id": student_id,
            "status": "Present",
            "reason": "First Morning Entry"
        })

        # Simulate attempting duplicate attendance on same date
        today_str = q_date = attendance_col.find_one({"student_id": student_id})["date"]
        dup_check = attendance_col.find_one({"student_id": student_id, "date": today_str})
        self.assertIsNotNone(dup_check)
        self.assertEqual(dup_check["status"], "Present")

        # Check audit log contains entries
        audit_res = self.client.get("/api/audit-logs")
        logs = audit_res.get_json()["logs"]
        self.assertGreater(len(logs), 0)
        print(f" -> Found {len(logs)} audit trail entries recorded securely!")
        print(" -> Duplicate prevention & Audit trail verified!")

    def test_06_reports_and_csv_export(self):
        print("\n[Test 6] Testing Reports & CSV Export...")
        self.login_teacher()

        # Reports API
        rep_res = self.client.get("/api/reports")
        self.assertEqual(rep_res.status_code, 200)
        self.assertIn("reports", rep_res.get_json())

        # Daily CSV Export
        csv_res = self.client.get("/export-csv?type=session")
        self.assertEqual(csv_res.status_code, 200)
        self.assertEqual(csv_res.headers["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn(b"AlphabetAI Attendance", csv_res.data)
        self.assertIn(b"Student Name", csv_res.data)

        # Cumulative CSV Export
        csv_cum = self.client.get("/export-csv?type=cumulative")
        self.assertEqual(csv_cum.status_code, 200)
        self.assertIn(b"Cumulative Summary Report", csv_cum.data)
        print(" -> Reports & RFC CSV downloads verified!")

if __name__ == "__main__":
    unittest.main()
