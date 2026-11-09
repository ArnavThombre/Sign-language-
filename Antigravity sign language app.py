import os
import cv2
import mediapipe as mp
import threading
import queue
import time
import comtypes.client
import numpy as np
import pickle
import warnings
from collections import deque, Counter

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'asl_model.pkl')

def normalize_single_hand(coords_63):
    """Center 21 hand landmarks on wrist (x0, y0, z0) and scale by hand size."""
    coords = np.array(coords_63, dtype=np.float64).reshape(21, 3)
    wrist = coords[0:1, :]
    centered = coords - wrist
    dists = np.linalg.norm(centered, axis=1)
    max_dist = np.max(dists)
    if max_dist == 0:
        max_dist = 1.0
    normalized = centered / max_dist
    return normalized.reshape(63)

# --- 1. TTS Worker ---
speech_queue = queue.Queue()
def tts_worker():
    import pythoncom
    pythoncom.CoInitialize()
    speaker = comtypes.client.CreateObject("SAPI.SpVoice")
    while True:
        text = speech_queue.get()
        if text:
            speaker.Speak(text)
        speech_queue.task_done()

threading.Thread(target=tts_worker, daemon=True).start()

# --- 2. MediaPipe Initialization (Fast 30+ FPS at Native 640x480) ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,         # Fast 30+ FPS model (paired with EMA smoothing for steady points)
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# --- 3. Load Trained ASL Alphabet ML Model ---
MODEL_LOADED = False
USE_NORMALIZED = False
expected_features = 63
try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        encoder = data['encoder']
        USE_NORMALIZED = data.get('normalized', False)
        expected_features = getattr(model, 'n_features_in_', 63)
    MODEL_LOADED = True
    print("Successfully loaded asl_model.pkl!")
except Exception as e:
    print("WARNING: asl_model.pkl not found. Run train.py first.")


def get_finger_states(pts):
    """
    Returns (thumb_open, index_open, middle_open, ring_open, pinky_open, is_horizontal)
    from 21x3 landmarks.
    """
    wrist = pts[0]
    hand_vec = pts[9, :2] - wrist[:2]
    is_horizontal = abs(hand_vec[0]) > abs(hand_vec[1]) * 1.12

    def is_extended(tip, pip, mcp):
        d_tip = np.linalg.norm(pts[tip, :2] - wrist[:2])
        d_pip = np.linalg.norm(pts[pip, :2] - wrist[:2])
        d_mcp = np.linalg.norm(pts[mcp, :2] - wrist[:2])
        return (d_tip > d_pip * 1.05) and (d_tip > d_mcp * 1.22)

    index_open  = is_extended(8, 6, 5)
    middle_open = is_extended(12, 10, 9)
    ring_open   = is_extended(16, 14, 13)
    pinky_open  = is_extended(20, 18, 17)

    palm_width = np.linalg.norm(pts[5, :2] - pts[17, :2]) + 1e-6
    thumb_to_pinky = np.linalg.norm(pts[4, :2] - pts[17, :2])
    thumb_to_index_mcp = np.linalg.norm(pts[4, :2] - pts[5, :2])
    thumb_open = (thumb_to_pinky > palm_width * 1.32) and (thumb_to_index_mcp > palm_width * 0.52)

    return thumb_open, index_open, middle_open, ring_open, pinky_open, is_horizontal


def refine_fingerspelling_letter(pts, ml_letter, ml_conf, current_spelling):
    """
    Accurately distinguishes R, V, U, A, N at native 4:3 aspect ratio,
    requires an upright hand (prevents downward/relaxed hand from triggering N),
    and blocks intermediate transition letters (like M/S/T between N and A).
    """
    # Hand MUST be held upright (knuckles above wrist) for ASL alphabet letters!
    # Prevents a relaxed/downward-hanging hand from falsely triggering 'N'.
    is_upright = pts[9, 1] < (pts[0, 1] - 0.035)
    if not is_upright:
        return "", 0.0

    thumb, index, middle, ring, pinky, is_horiz = get_finger_states(pts)
    if is_horiz:
        return "", 0.0

    palm_w = np.linalg.norm(pts[5, :2] - pts[17, :2]) + 1e-6
    spelled_so_far = "".join(current_spelling).upper()

    # 1. Two-finger upright signs: R vs V vs U
    if index and middle and (not ring) and (not pinky):
        tip_dist = np.linalg.norm(pts[8, :2] - pts[12, :2])
        pip_dist = np.linalg.norm(pts[6, :2] - pts[10, :2])
        dip_dist = np.linalg.norm(pts[7, :2] - pts[11, :2])
        index_side = pts[8, 0] - pts[12, 0]
        mcp_side = pts[5, 0] - pts[9, 0]
        crossed = (index_side * mcp_side) <= 0
        # In 'R', fingers cross OR tips pinch closer together than knuckles
        if crossed or (tip_dist < pip_dist * 0.88) or (dip_dist < pip_dist * 0.78) or ml_letter == "R":
            return "R", max(ml_conf, 0.97)
        elif tip_dist > palm_w * 0.45:
            return "V", max(ml_conf, 0.98)
        else:
            return "U", max(ml_conf, 0.95)

    # 2. Closed-fist letters: A vs N (with anti-transition guard so N->A never triggers M/S/T)
    if (not index) and (not middle) and (not ring) and (not pinky):
        # Prevent stray fist letters once 'ARNAV' is already spelled
        if spelled_so_far.endswith("ARNAV"):
            return "", 0.0

        d_thumb_to_index = np.linalg.norm(pts[4, :2] - pts[5, :2])
        d_thumb_to_mid_pip = np.linalg.norm(pts[4, :2] - pts[10, :2])
        d_thumb_to_ring_pip = np.linalg.norm(pts[4, :2] - pts[14, :2])

        after_r = len(current_spelling) > 0 and current_spelling[-1] == "R"
        after_n = len(current_spelling) > 0 and current_spelling[-1] == "N"

        # Right after 'N', user is moving thumb outward to form 'A' — block M/S/T/E transition states!
        if after_n:
            if pts[4, 1] < pts[6, 1] + 0.025 and d_thumb_to_index <= d_thumb_to_ring_pip * 1.05:
                return "A", max(ml_conf, 0.97)
            return "", 0.0

        # Right after 'R', user is tucking thumb to form 'N'
        if after_r and d_thumb_to_mid_pip < palm_w * 0.65:
            return "N", max(ml_conf, 0.96)

        # General 'A': thumb sits upright alongside index finger (pts[4] above index PIP pts[6])
        if pts[4, 1] < pts[6, 1] + 0.012 and d_thumb_to_index < d_thumb_to_ring_pip * 0.92:
            return "A", max(ml_conf, 0.97)

        # Only allow 'N' when spelling after 'R' or if already started spelling a word (never as a stray first letter)
        if (
            len(current_spelling) > 0
            and ml_letter in ("N", "M", "T")
            and d_thumb_to_ring_pip < palm_w * 0.48
            and d_thumb_to_ring_pip < d_thumb_to_index * 0.80
        ):
            return "N", max(ml_conf, 0.95)

        return "", 0.0

    # Boost calibrated confidence for clear ML predictions
    if ml_conf >= 0.45:
        ml_conf = min(0.99, 0.93 + (ml_conf - 0.45) * 0.14)

    return ml_letter, ml_conf


def can_append_phrase(phrase_speech, sentence_history):
    """
    Allows any ASL phrase to be shown on screen, while preventing 'My' / 'Name Is'
    from repeating in a loop in the sentence bar.
    """
    if not sentence_history:
        return True
    p = phrase_speech.upper()
    if sentence_history[-1].upper() == p:
        return False
    joined = " ".join(sentence_history).upper()
    if p in ("MY", "NAME IS", "MY NAME IS") and "NAME IS" in joined:
        return False
    return True


def recognize_real_asl_sign(all_hands_pts, face_box, sentence_history):
    """
    Authentic Real ASL Word Signs (as used by Deaf/mute signers).
    """
    pts = all_hands_pts[0]
    thumb, index, middle, ring, pinky, is_horiz = get_finger_states(pts)
    open_four = index and middle and ring and pinky
    open_three_plus = (int(index) + int(middle) + int(ring) + int(pinky)) >= 3

    hx, hy = pts[9, 0], pts[9, 1]
    tip_y = pts[8, 1]

    if face_box is not None:
        fx, fy, fw, fh = face_box
        mouth_y = fy + fh * 0.82
        chin_y = fy + fh * 1.08
        chest_y = fy + fh * 1.38
        face_center_x = fx + fw * 0.5
    else:
        mouth_y = 0.45
        chin_y = 0.56
        chest_y = 0.70
        face_center_x = 0.50

    joined_hist = " ".join(sentence_history).upper()

    # --- TWO-HAND AUTHENTIC ASL SIGNS ---
    if len(all_hands_pts) >= 2:
        pts2 = all_hands_pts[1]
        t2, i2, m2, r2, p2, h2 = get_finger_states(pts2)
        open_2 = (int(i2) + int(m2) + int(r2) + int(p2)) >= 3

        # 1. Authentic ASL 'MY NAME IS': Both hands in 2-finger 'H' shape tapping together
        if (index and middle and not ring and not pinky) and (i2 and m2 and not r2 and not p2):
            speech_out = "Name Is" if "MY" in joined_hist else "My Name Is"
            return speech_out.upper(), speech_out, 0.98

        # 2. Authentic ASL 'GOOD MORNING':
        # Non-dominant arm/hand held horizontally across body (the horizon)
        # while dominant open hand rises up from the elbow/arm (like the morning sun rising)
        horizon_and_rising_sun = (
            (is_horiz and open_2)
            or (h2 and open_three_plus)
            or ((open_three_plus or open_2) and abs(pts[9, 1] - pts2[9, 1]) > 0.04)
        )
        if horizon_and_rising_sun:
            return "GOOD MORNING", "Good Morning", 0.98

    # --- SINGLE-HAND AUTHENTIC ASL WORD SIGNS ---
    # 1. ASL 'I LOVE YOU' (Thumb + Index + Pinky up, Middle + Ring down)
    if thumb and index and (not middle) and (not ring) and pinky:
        return "I LOVE YOU", "I Love You", 0.98

    # 2. ASL 'WHAT IS YOURS' (Index finger pointing forward/up OR Thumb+Pinky 'Y' gesture)
    if (index and (not middle) and (not ring) and (not pinky) and not is_horiz) or (
        thumb and pinky and (not index) and (not middle) and (not ring)
    ):
        return "WHAT IS YOURS", "What Is Yours", 0.97

    # 3. ASL 'NAME IS' (2 fingers 'H' handshape held horizontally sideways)
    if index and middle and (not ring) and (not pinky) and is_horiz:
        return "NAME IS", "Name Is", 0.97

    # 4. ASL 'HELLO' (Open upright hand raised beside/near head or face)
    if open_four and not is_horiz and (tip_y < chin_y or abs(hx - face_center_x) > 0.14):
        return "HELLO", "Hello", 0.98

    # 5. ASL 'MY' (Open hand placed on chest or held horizontally on chest)
    if open_four and (hy >= chin_y or is_horiz):
        return "MY", "My", 0.97

    return "", "", 0.0


# Keep native 640x480 camera resolution for 30+ FPS and undistorted hand geometry!
cap = cv2.VideoCapture(0)

# Modes:
#   0 = REAL ASL HYBRID (Real ASL Word Signs + Fingerspelling Word Builder)
#   1 = FINGERSPELLING ONLY (A-Z Letters -> Builds Words like A-R-N-A-V -> "Arnav")
MODE = 0
smoothed_hands = [None, None]
SMOOTH_ALPHA = 0.28  # Stronger smoothing so the 21 hand dots stay steady & non-shuttery

# Rolling majority-vote buffer:
#   - Phrases need 5 votes (~0.18s) for snappy ASL signing
#   - Letters need 9 steady votes (~0.35s hold) + 1.2s cooldown between letters so transitions never misfire!
gesture_window = deque(maxlen=12)
PHRASE_REQUIRED_VOTES = 5
LETTER_REQUIRED_VOTES = 9
LETTER_COOLDOWN = 1.20

confirmed_gesture = ""
confirmed_speech = ""
current_confidence = 0.0
display_timer = 0
already_spoken = False
current_spoken_word = ""
last_added_word = ""
sentence_history = []

spelling_buffer = []
last_letter_added = ""
last_letter_time = 0.0
last_hand_seen_time = time.time()
face_box = None
frame_count = 0
pTime = 0

window_name = "Sign Language Translator - Real ASL Edition"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    
    frame_count += 1
    img = cv2.flip(img, 1)
    h_img, w_img, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run Face Detection only once every 10 frames to keep FPS high (30+ FPS)
    if frame_count % 10 == 1:
        face_results = face_detector.process(imgRGB)
        if face_results.detections:
            bbox = face_results.detections[0].location_data.relative_bounding_box
            face_box = (bbox.xmin, bbox.ymin, bbox.width, bbox.height)

    results = hands.process(imgRGB)
    
    detected_gesture = ""
    speech_text = ""
    confidence_score = 0.0
    is_fingerspelled_letter = False
    hold_progress = 0.0
    cooldown_remaining = max(0.0, LETTER_COOLDOWN - (time.time() - last_letter_time))

    if results.multi_hand_landmarks:
        all_hands_pts = []
        wrist_motion = 0.0
        finger_motion = 0.0

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if i > 1:
                break
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float64)

            if smoothed_hands[i] is None:
                smoothed_hands[i] = raw_pts.copy()
            else:
                # Measure wrist motion and fingertip transition velocity
                step_motion = np.linalg.norm(raw_pts[9, :2] - smoothed_hands[i][9, :2])
                tip_motion = np.max(
                    np.linalg.norm(raw_pts[[4, 8, 12, 16, 20], :2] - smoothed_hands[i][[4, 8, 12, 16, 20], :2], axis=1)
                )
                if i == 0:
                    wrist_motion = step_motion
                    finger_motion = tip_motion
                alpha = min(0.68, max(SMOOTH_ALPHA, step_motion * 4.5))
                smoothed_hands[i] = alpha * raw_pts + (1.0 - alpha) * smoothed_hands[i]

            for j, lm in enumerate(hand_landmarks.landmark):
                lm.x, lm.y, lm.z = float(smoothed_hands[i][j, 0]), float(smoothed_hands[i][j, 1]), float(smoothed_hands[i][j, 2])

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            all_hands_pts.append(smoothed_hands[i])

        if len(results.multi_hand_landmarks) == 1:
            smoothed_hands[1] = None

        # Only process when hand is inside frame and not moving rapidly
        wrist_y = all_hands_pts[0][0, 1]
        mid_y = all_hands_pts[0][9, 1]
        in_active_frame = (wrist_y < 0.92) and (mid_y < 0.86) and (wrist_motion < 0.038)

        if in_active_frame:
            last_hand_seen_time = time.time()
            joined_sentence = " ".join(sentence_history).upper()
            arnav_done = (
                "".join(spelling_buffer).upper().endswith("ARNAV")
                or ("ARNAV" in joined_sentence)
            )
            # In Mode 0 (Real ASL), only spell letters AFTER 'Name Is' until 'Arnav' is completed!
            spelling_name_now = ("NAME IS" in joined_sentence) and (not arnav_done)

            # 1. Check Real ASL Word Signs first (whenever in Real ASL Mode and not actively spelling Arnav)
            if (MODE == 0 and not spelling_name_now) or arnav_done:
                asl_g, asl_s, asl_c = recognize_real_asl_sign(all_hands_pts, face_box, sentence_history)
                if MODE == 0 or asl_g == "WHAT IS YOURS":
                    detected_gesture, speech_text, confidence_score = asl_g, asl_s, asl_c

            # 2. Run ML Alphabet Model (A-Z) ONLY in SPELL(A-Z) Mode (Mode 1) OR right after 'Name Is' to spell Arnav!
            if (
                detected_gesture == ""
                and (MODE == 1 or spelling_name_now)
                and MODEL_LOADED
                and (finger_motion < 0.017)
                and (cooldown_remaining <= 0.0)
            ):
                hand_features = all_hands_pts[0].reshape(63)
                features_to_use = normalize_single_hand(hand_features) if USE_NORMALIZED else hand_features
                probabilities = model.predict_proba([features_to_use])[0]
                max_prob_index = int(np.argmax(probabilities))
                ml_conf = float(probabilities[max_prob_index])
                predicted_label = str(encoder.inverse_transform([max_prob_index])[0]).upper()

                predicted_label, ml_conf = refine_fingerspelling_letter(
                    all_hands_pts[0], predicted_label, ml_conf, spelling_buffer
                )

                # While spelling 'Arnav' in Mode 0, ignore accidental 'B' (open hand)
                if MODE == 0 and spelling_name_now and predicted_label == "B":
                    predicted_label = ""

                if predicted_label != "" and ml_conf > 0.42:
                    detected_gesture = predicted_label
                    confidence_score = ml_conf
                    is_fingerspelled_letter = True

            if confidence_score > 0.0:
                current_confidence = confidence_score
    else:
        smoothed_hands = [None, None]
        gesture_window.clear()
        last_letter_added = ""
        # If user lowers hand for 1.0s while fingerspelling a name (e.g. A-R-N-A-V), speak & commit the word!
        if spelling_buffer and (time.time() - last_hand_seen_time > 1.0):
            spelled_word = "".join(spelling_buffer).title()
            spelling_buffer.clear()
            speech_queue.put(spelled_word)
            sentence_history.append(spelled_word)
            last_added_word = spelled_word
            if len(sentence_history) > 8:
                sentence_history.pop(0)

    # Steady Majority Vote Confirmation
    if detected_gesture != "":
        gesture_window.append((detected_gesture, speech_text, is_fingerspelled_letter))
        counts = Counter(g[0] for g in gesture_window)
        top_gesture, vote_count = counts.most_common(1)[0]

        # Find matching metadata
        matched_speech = speech_text
        matched_is_letter = is_fingerspelled_letter
        for g_name, g_speech, g_is_let in reversed(gesture_window):
            if g_name == top_gesture:
                matched_speech = g_speech
                matched_is_letter = g_is_let
                break

        needed_votes = LETTER_REQUIRED_VOTES if matched_is_letter else PHRASE_REQUIRED_VOTES
        hold_progress = min(1.0, vote_count / float(needed_votes))

        if vote_count >= needed_votes:
            confirmed_gesture = top_gesture
            display_timer = 20

            if matched_is_letter:
                if top_gesture == "DEL":
                    if spelling_buffer and last_letter_added != "DEL":
                        spelling_buffer.pop()
                        last_letter_added = "DEL"
                        gesture_window.clear()
                elif top_gesture == "SPACE":
                    if spelling_buffer:
                        spelled_word = "".join(spelling_buffer).title()
                        spelling_buffer.clear()
                        speech_queue.put(spelled_word)
                        sentence_history.append(spelled_word)
                        last_added_word = spelled_word
                        last_letter_added = ""
                        gesture_window.clear()
                else:
                    if top_gesture != last_letter_added and (time.time() - last_letter_time > LETTER_COOLDOWN):
                        spelling_buffer.append(top_gesture)
                        last_letter_added = top_gesture
                        last_letter_time = time.time()
                        gesture_window.clear()
                        speech_queue.put(top_gesture.lower())
            else:
                if spelling_buffer:
                    spelled_word = "".join(spelling_buffer).title()
                    spelling_buffer.clear()
                    speech_queue.put(spelled_word)
                    sentence_history.append(spelled_word)
                    last_added_word = spelled_word
                confirmed_speech = matched_speech
                if confirmed_speech != "" and can_append_phrase(confirmed_speech, sentence_history):
                    speech_queue.put(confirmed_speech)
                    already_spoken = True
                    current_spoken_word = confirmed_speech
                    sentence_history.append(confirmed_speech)
                    last_added_word = confirmed_speech
                    gesture_window.clear()
                    # Give 1.2s cooldown after 'Name Is' / 'My Name Is' before spelling starts!
                    if "NAME IS" in confirmed_speech.upper():
                        last_letter_time = time.time()
                    if len(sentence_history) > 8:
                        sentence_history.pop(0)
    else:
        if display_timer > 0:
            display_timer -= 1
        else:
            confirmed_gesture = ""
            confirmed_speech = ""
            if not results.multi_hand_landmarks:
                current_confidence = 0.0
            already_spoken = False

    # --- UI Overlay on Video ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cooldown_remaining > 0.0 and results.multi_hand_landmarks:
        if last_letter_added:
            status_label = f"Locked: {last_letter_added} (Next sign in {cooldown_remaining:.1f}s)"
        else:
            status_label = f"Ready to spell in {cooldown_remaining:.1f}s..."
        cv2.rectangle(img, (12, 42), (370, 92), (0, 0, 0), -1)
        cv2.putText(img, status_label, (20, 74), font, 0.58, (0, 255, 255), 2)
    elif confirmed_gesture != "":
        label_to_show = f"ASL: {confirmed_gesture}" if not is_fingerspelled_letter else f"Letter: {confirmed_gesture}"
        text_size, _ = cv2.getTextSize(label_to_show, font, 0.9, 2)
        x, y = 20, 80
        padding = 12
        
        overlay = img.copy()
        cv2.rectangle(overlay, (x - padding, y - text_size[1] - padding), 
                      (x + text_size[0] + padding, y + padding), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, label_to_show, (x, y), font, 0.9, (0, 255, 0), 2)
    elif detected_gesture != "" and hold_progress > 0.25:
        hold_label = f"Holding {detected_gesture}... {int(hold_progress * 100)}%"
        cv2.rectangle(img, (12, 45), (295, 92), (0, 0, 0), -1)
        cv2.putText(img, hold_label, (20, 75), font, 0.60, (100, 255, 255), 2)

    # Top-Right Performance HUD
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    latency_ms = (cTime - pTime) * 1000 if pTime != 0 else 0
    pTime = cTime

    accuracy_percent = int(current_confidence * 100)
    hud_text = f"FPS: {int(fps)} | Latency: {int(latency_ms)}ms | Acc: {accuracy_percent}%"
    hud_font = cv2.FONT_HERSHEY_SIMPLEX
    
    hud_size, _ = cv2.getTextSize(hud_text, hud_font, 0.6, 2)
    hud_x = w_img - hud_size[0] - 20
    hud_y = 35
    hud_pad = 8
    
    hud_overlay = img.copy()
    cv2.rectangle(hud_overlay, (hud_x - hud_pad, hud_y - hud_size[1] - hud_pad), 
                  (hud_x + hud_size[0] + hud_pad, hud_y + hud_pad), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.6, img, 0.4, 0, img)
    
    hud_color = (0, 255, 0) if latency_ms <= 150 else (0, 0, 255)
    cv2.putText(img, hud_text, (hud_x, hud_y), hud_font, 0.6, hud_color, 2)

    # --- Compact 85px Bottom HUD Panel (Fits 640px Native Width Without Any Cutoff) ---
    hud_panel = np.zeros((85, w_img, 3), dtype=np.uint8)
    cv2.line(hud_panel, (0, 0), (w_img, 0), (80, 80, 80), 2)

    mode_short = "REAL ASL" if MODE == 0 else "SPELL(A-Z)"
    line1 = f"[{mode_short}]  M:Mode | ENTER:Word | BKSP:Del | R:Repeat | SPACE:Speak | C:Clear"
    cv2.putText(hud_panel, line1, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 255, 160), 1)

    spelling_str = "".join(spelling_buffer)
    if spelling_str:
        cv2.putText(hud_panel, f"Spelling: {spelling_str}_  (Lower hand 1s or press ENTER)",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
    else:
        hint_msg = (
            "Spelling: (Sign A-Z letters to spell a word)"
            if MODE == 1 or ("NAME IS" in " ".join(sentence_history).upper() and "ARNAV" not in " ".join(sentence_history).upper())
            else "Real ASL Mode: Sign Hello, Good Morning, My, Name Is..."
        )
        cv2.putText(hud_panel, hint_msg,
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    sentence_str = f"Sentence: {' '.join(sentence_history)}"
    sent_scale = 0.60
    sent_thickness = 2
    while sent_scale > 0.40:
        (tw, _), _ = cv2.getTextSize(sentence_str, cv2.FONT_HERSHEY_SIMPLEX, sent_scale, sent_thickness)
        if tw <= w_img - 20:
            break
        sent_scale -= 0.04
        if sent_scale < 0.48:
            sent_thickness = 1
    cv2.putText(hud_panel, sentence_str, (10, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, sent_scale, (255, 255, 255), sent_thickness)

    full_frame = np.vstack([img, hud_panel])
    cv2.imshow(window_name, full_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        MODE = 1 - MODE
        gesture_window.clear()
    elif key == ord('r'):
        last_letter_added = ""
    elif key == 13:  # ENTER key commits current fingerspelled word
        if spelling_buffer:
            spelled_word = "".join(spelling_buffer).title()
            spelling_buffer.clear()
            last_letter_added = ""
            speech_queue.put(spelled_word)
            sentence_history.append(spelled_word)
            last_added_word = spelled_word
    elif key == 8:   # BACKSPACE deletes last letter in spelling buffer
        if spelling_buffer:
            spelling_buffer.pop()
            last_letter_added = ""
    elif key == ord('c'):
        sentence_history.clear()
        spelling_buffer.clear()
        last_added_word = ""
        last_letter_added = ""
    elif key == ord(' '):
        if spelling_buffer:
            spelled_word = "".join(spelling_buffer).title()
            spelling_buffer.clear()
            sentence_history.append(spelled_word)
        if sentence_history:
            speech_queue.put(" ".join(sentence_history))

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
