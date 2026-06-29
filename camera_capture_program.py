"""
Smart Motion Detection + Face Recognition Camera Program
====================================
Features:
- Motion detection → Human detection → Face recognition, three-layer filtering
- Auto-ignore known people, capture photo + alarm for strangers
- Can't see face (back turned / head down) → treated as unknown, always capture photo
- Photos have date/time watermark in top-left corner
- Photos saved to desktop 'pic' folder (strangers/unknown stored separately)

Usage:
    pip install opencv-python face_recognition numpy

    # Step 1: Register your face (sit in front of the camera)
    python motion_capture.py --register your_name

    # Step 2: Start monitoring
    python motion_capture.py

Press 'q' to exit the program.

V2 Updates:
- Added human detection, only captures photos when a "person" is detected (ignores cats, curtains blowing, etc.)
- Face recognition distinguishes "known people" from "strangers", only strangers trigger alarms

V3 Updates:
- Previously only full body detection, now supports upper body, face, and profile detection

"""

import cv2
import numpy as np
import time
import os
import sys
import platform
import pickle
import argparse
from datetime import datetime

try:
    import face_recognition
    FACE_REC_AVAILABLE = True
except ImportError:
    FACE_REC_AVAILABLE = False


# ============ Adjustable Parameters ============

MOTION_THRESHOLD = 2000       # Motion area threshold (lower = more sensitive)
DELAY_BEFORE_PHOTO = 1.0      # Seconds to wait after motion detected before taking photo
COOLDOWN_AFTER_PHOTO = 5.0    # Seconds to wait after photo before taking another
CAMERA_INDEX = 0              # Camera index (0 = default)
SHOW_PREVIEW = False           # Whether to show preview window
FACE_MATCH_TOLERANCE = 0.5    # Face match tolerance (lower = stricter, recommended 0.4-0.6)
REGISTER_PHOTO_COUNT = 10     # Number of photos to take during registration
ALARM_SOUND = False            # Whether to sound alarm for strangers

# =========================================


def get_desktop_path():
    """Get the desktop path."""
    system = platform.system()
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    if system == "Linux" and not os.path.exists(desktop):
        desktop = os.path.expanduser("~")
    return desktop


def get_save_folders():
    """Create 'pic' folder and subfolders on desktop."""
    desktop = get_desktop_path()
    base = os.path.join(desktop, "pic")
    strangers = os.path.join(base, "strangers")   # Strangers
    unknown = os.path.join(base, "unknown")        # Can't see face
    for folder in [base, strangers, unknown]:
        os.makedirs(folder, exist_ok=True)
    return base, strangers, unknown


def get_known_faces_path():
    """Get file path for stored registered face data."""
    desktop = get_desktop_path()
    data_dir = os.path.join(desktop, "pic", "face_data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "known_faces.pkl")


def load_known_faces():
    """Load registered face data."""
    path = get_known_faces_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data['names'])} registered face encodings")
        # Show registered people
        unique_names = sorted(set(data["names"]))
        print(f"   Registered: {', '.join(unique_names)}")
        return data["encodings"], data["names"]
    return [], []


def save_known_faces(encodings, names):
    """Save face data to file."""
    path = get_known_faces_path()
    with open(path, "wb") as f:
        pickle.dump({"encodings": encodings, "names": names}, f)
    print(f"💾 Face data saved to: {path}")


# ============================================================
#  Registration Mode: Sit in front of the camera, program
#  automatically captures your face and records encodings
# ============================================================
def register_face(name):
    """Register a new known face."""
    if not FACE_REC_AVAILABLE:
        print("❌ face_recognition library required:")
        print("   pip install face_recognition")
        return

    print(f"\n👤 Starting face registration: {name}")
    print(f"   Program will capture {REGISTER_PHOTO_COUNT} photos of your face")
    print("   Please face the camera, you can slightly turn your head for different angles")
    print("   Press 'q' to finish early\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    # Load existing data
    encodings, names = load_known_faces()

    captured = 0
    last_capture = 0

    while captured < REGISTER_PHOTO_COUNT:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()

        # Downscale frame for faster detection
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small)

        if face_locations and (current_time - last_capture) > 1.0:
            # Extract face encodings
            face_encs = face_recognition.face_encodings(rgb_small, face_locations)
            if face_encs:
                encodings.append(face_encs[0])
                names.append(name)
                captured += 1
                last_capture = current_time
                print(f"   📸 [{captured}/{REGISTER_PHOTO_COUNT}] Captured")

        # Preview
        for (top, right, bottom, left) in face_locations:
            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f"Registering: {name} ({captured}/{REGISTER_PHOTO_COUNT})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Face the camera, change angles slowly",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured > 0:
        save_known_faces(encodings, names)
        print(f"\n✅ Registration complete! Recorded {captured} face encodings for '{name}'.")
    else:
        print("\n⚠️  No faces captured, please try again.")


# ============================================================
#  Human Body Detector (upper body + face + full body,
#  any hit counts as "person detected")
# ============================================================
def create_body_detector():
    """Create multi-method human body detector."""
    detectors = {}

    # Upper body detector (works even when sitting)
    upper_path = cv2.data.haarcascades + "haarcascade_upperbody.xml"
    detectors["upperbody"] = cv2.CascadeClassifier(upper_path)

    # Face detector (frontal face)
    face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detectors["face"] = cv2.CascadeClassifier(face_path)

    # Face detector (profile/side face)
    profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"
    detectors["profile"] = cv2.CascadeClassifier(profile_path)

    # Full body detector (standing person)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    detectors["fullbody_hog"] = hog

    return detectors


def detect_bodies(detectors, frame):
    """Detect whether there are people in the frame (upper body/face/full body, any match counts). Returns list of detection boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_detections = []

    # Upper body detection (most effective for seated people)
    upper = detectors["upperbody"].detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60)
    )
    for (x, y, w, h) in upper:
        all_detections.append((x, y, w, h, "upperbody"))

    # Frontal face detection
    faces = detectors["face"].detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    for (x, y, w, h) in faces:
        all_detections.append((x, y, w, h, "face"))

    # Profile/side face detection
    profiles = detectors["profile"].detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
    )
    for (x, y, w, h) in profiles:
        all_detections.append((x, y, w, h, "profile"))

    # Full body HOG detection
    small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    bodies, _ = detectors["fullbody_hog"].detectMultiScale(
        small, winStride=(8, 8), padding=(4, 4), scale=1.05
    )
    for (x, y, w, h) in bodies:
        all_detections.append((x * 2, y * 2, w * 2, h * 2, "fullbody"))

    return all_detections


# ============================================================
#  Photo Watermark
# ============================================================
def add_timestamp(frame, label=""):
    """Add date/time watermark to top-left corner of photo."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"{ts}  {label}" if label else ts
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


# ============================================================
#  Alarm
# ============================================================
def trigger_alarm(label):
    """Trigger alarm (terminal alert + system sound)."""
    print(f"🚨 ALARM! Detected: {label}")
    if ALARM_SOUND:
        # Cross-platform attempt to play alert sound
        try:
            if platform.system() == "Windows":
                import winsound
                winsound.Beep(1000, 500)
            elif platform.system() == "Darwin":
                os.system("afplay /System/Library/Sounds/Sosumi.aiff &")
            else:
                print("\a")  # Terminal beep
        except Exception:
            print("\a")


# ============================================================
#  Main Monitoring Loop
# ============================================================
def monitor():
    """Main monitoring mode."""
    base_folder, strangers_folder, unknown_folder = get_save_folders()

    # Load human body detectors
    detectors = create_body_detector()
    print("✅ Human body detectors loaded (upper body + face + full body)")

    # Load face recognition data
    known_encodings, known_names = [], []
    if FACE_REC_AVAILABLE:
        known_encodings, known_names = load_known_faces()
        if not known_encodings:
            print("⚠️  No known faces registered yet! Everyone will be treated as a stranger.")
            print("   Use --register name to register known people\n")
    else:
        print("⚠️  face_recognition not installed, will only do human body detection (no known/stranger distinction)")
        print("   Install: pip install face_recognition\n")

    # Open camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    print("✅ Camera opened, starting smart monitoring...")
    print("   Press 'q' to exit\n")
    print("   Logic: Motion → Person? → Face? → Known? → Headcount matches?")
    print("   ├─ No human body → Ignore (cat/wind/shadows)")
    print("   ├─ Person but no face → Capture (possibly back turned)")
    print("   ├─ All faces known & headcount matches → Ignore")
    print("   ├─ Known + someone without visible face → Capture (unidentified person)")
    print("   └─ Stranger face → Capture + Alarm\n")

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("❌ Cannot read camera feed.")
        cap.release()
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)

    # State variables
    motion_detected_time = None
    last_photo_time = 0
    photo_count = 0
    waiting_to_capture = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.5)
            continue

        current_time = time.time()
        frame_count += 1

        # Convert to grayscale + blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Compute frame difference
        frame_delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_now = any(cv2.contourArea(c) > MOTION_THRESHOLD for c in contours)
        in_cooldown = (current_time - last_photo_time) < COOLDOWN_AFTER_PHOTO

        # ---- State Machine ----
        if motion_now and not in_cooldown and not waiting_to_capture:
            motion_detected_time = current_time
            waiting_to_capture = True
            print("🔍 Motion detected, analyzing...")

        elif waiting_to_capture:
            elapsed = current_time - motion_detected_time
            if elapsed >= DELAY_BEFORE_PHOTO:
                # ---- Delay elapsed, start analyzing this frame ----

                # Layer 1: Human body detection (upper body/face/full body, any hit)
                bodies = detect_bodies(detectors, frame)

                if not bodies:
                    # No human body → probably cat/wind → ignore
                    print("   🐱 No human body detected, ignoring (likely animal/shadows)")
                    waiting_to_capture = False
                    motion_detected_time = None
                else:
                    # Print detected types
                    det_types = set(b[4] for b in bodies)
                    # Use non-face detectors to estimate body count (upperbody + fullbody)
                    body_only = [b for b in bodies if b[4] in ("upperbody", "fullbody")]
                    num_bodies = max(len(body_only), 1)  # At least count as 1 person
                    print(f"   👤 Detected approximately {num_bodies} human bodies (methods: {', '.join(det_types)})")

                    # Person detected → Layer 2: Face detection
                    face_label = None       # Final label
                    save_folder = None      # Which folder to save to

                    if FACE_REC_AVAILABLE:
                        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                        face_locations = face_recognition.face_locations(rgb_small)

                        num_faces = len(face_locations)
                        print(f"   🔎 Detected {num_faces} faces, {num_bodies} bodies")

                        if face_locations:
                            # Faces found → Layer 3: Who is it?
                            face_encs = face_recognition.face_encodings(rgb_small, face_locations)

                            known_count = 0
                            stranger_count = 0

                            for enc in face_encs:
                                if known_encodings:
                                    distances = face_recognition.face_distance(known_encodings, enc)
                                    min_dist = min(distances)
                                    best_idx = np.argmin(distances)

                                    if min_dist < FACE_MATCH_TOLERANCE:
                                        matched_name = known_names[best_idx]
                                        known_count += 1
                                        print(f"   👋 Known person: {matched_name} (match confidence: {1-min_dist:.1%})")
                                    else:
                                        stranger_count += 1
                                        print(f"   ⚠️  Stranger detected! (closest match distance: {min_dist:.2f})")
                                else:
                                    stranger_count += 1

                            # Key: Compare body count vs face count
                            # If more bodies than faces → someone has their back turned, don't let them pass
                            unidentified = num_bodies - num_faces
                            if unidentified > 0:
                                print(f"   🔄 {unidentified} bodies without matching faces (possibly back turned)")

                            if stranger_count > 0:
                                # Stranger face detected
                                face_label = "STRANGER"
                                save_folder = strangers_folder
                                for (top, right, bottom, left) in face_locations:
                                    top *= 2; right *= 2; bottom *= 2; left *= 2
                                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            elif unidentified > 0:
                                # All faces are known, but some bodies don't have faces → capture
                                face_label = "PARTIAL_ID"
                                save_folder = unknown_folder
                                print(f"   ⚠️  Known: {known_count} + Unidentified: {unidentified} → capturing photo")
                            else:
                                # All faces are known, headcount matches → safe
                                print(f"   ✅ All known people ({known_count}), skipping photo")
                                waiting_to_capture = False
                                motion_detected_time = None
                                prev_gray = gray.copy()
                                continue
                        else:
                            # Person detected but no face → possibly back turned
                            face_label = "NO_FACE"
                            save_folder = unknown_folder
                    else:
                        # face_recognition not installed → capture anyone
                        face_label = "PERSON"
                        save_folder = strangers_folder

                    # ---- Capture Photo ----
                    if face_label and save_folder:
                        photo_count += 1
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{face_label}_{timestamp}_{photo_count:04d}.jpg"
                        filepath = os.path.join(save_folder, filename)

                        # Draw body detection boxes
                        for (x, y, w, h, det_type) in bodies:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 2)
                            cv2.putText(frame, det_type, (x, y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

                        # Add watermark
                        add_timestamp(frame, face_label)

                        cv2.imwrite(filepath, frame)
                        last_photo_time = current_time

                        print(f"   📸 [{face_label}] Photo captured! -> {filename}")
                        print(f"   ⏳ Cooldown {COOLDOWN_AFTER_PHOTO} seconds...\n")

                        # Trigger alarm for strangers
                        if face_label == "STRANGER":
                            trigger_alarm("Stranger!")

                    waiting_to_capture = False
                    motion_detected_time = None

        # Update previous frame
        prev_gray = gray.copy()

        # Preview window
        if SHOW_PREVIEW:
            if in_cooldown:
                status, color = "COOLDOWN", (0, 0, 255)
            elif waiting_to_capture:
                status, color = "ANALYZING...", (0, 255, 255)
            else:
                status, color = "MONITORING", (0, 255, 0)

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Photos: {photo_count}", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            known_count = len(set(known_names)) if known_names else 0
            cv2.putText(frame, f"Known faces: {known_count}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            if in_cooldown:
                remaining = COOLDOWN_AFTER_PHOTO - (current_time - last_photo_time)
                cv2.putText(frame, f"Cooldown: {remaining:.1f}s", (10, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Smart Motion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🛑 Program ended, captured {photo_count} photos total.")
    print(f"📁 Stranger photos: {strangers_folder}")
    print(f"📁 Unknown (no face) photos: {unknown_folder}")


# ============================================================
#  Command Line Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Smart Motion Detection + Face Recognition Surveillance System")
    parser.add_argument("--register", type=str, metavar="NAME",
                        help="Register a known face (e.g.: --register John)")
    parser.add_argument("--list", action="store_true",
                        help="View list of registered known people")
    parser.add_argument("--delete", type=str, metavar="NAME",
                        help="Delete a known person's face data")
    args = parser.parse_args()

    if args.register:
        register_face(args.register)
    elif args.list:
        encodings, names = load_known_faces()
        if names:
            unique = sorted(set(names))
            print(f"\n{len(unique)} known people registered:")
            for n in unique:
                count = names.count(n)
                print(f"   👤 {n} ({count} encodings)")
        else:
            print("\nNo known people registered yet.")
            print("Use: python motion_capture.py --register name")
    elif args.delete:
        encodings, names = load_known_faces()
        if args.delete in names:
            new_enc = [e for e, n in zip(encodings, names) if n != args.delete]
            new_names = [n for n in names if n != args.delete]
            save_known_faces(new_enc, new_names)
            print(f"✅ Deleted all face data for '{args.delete}'")
        else:
            print(f"❌ '{args.delete}' not found")
    else:
        monitor()


if __name__ == "__main__":
    main()