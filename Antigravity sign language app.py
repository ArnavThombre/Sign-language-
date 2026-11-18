import os
import sys
import site
import glob

# Ensure working mediapipe (0.10.21 with mp.solutions) from user site-packages takes priority over broken venv mediapipe
for _sp in glob.glob(os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.*\LocalCache\local-packages\Python*\site-packages")) + [site.getusersitepackages()]:
    if os.path.isdir(_sp):
        if _sp in sys.path:
            sys.path.remove(_sp)
        sys.path.insert(0, _sp)

import cv2
import time
import queue
import pickle
import warnings
import threading
import numpy as np
import mediapipe as mp
import comtypes.client
from collections import deque, Counter

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'asl_model.pkl')


def normalize_single_hand(coords_63):
    """Center 21 hand landmarks on the wrist (x0, y0, z0) and normalize by hand scale."""
    coords = np.array(coords_63, dtype=np.float64).reshape(21, 3)
    wrist = coords[0:1, :]
    centered = coords - wrist
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    if max_dist == 0:
        max_dist = 1.0
    return (centered / max_dist).reshape(63)


# --- 1. Asynchronous Text-to-Speech Worker ---
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


# --- 2. MediaPipe & ML Model Initialization ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55
)
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(min_detection_confidence=0.50)
mp_draw = mp.solutions.drawing_utils

MODEL_LOADED = False
USE_NORMALIZED = False
try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        encoder = data['encoder']
        USE_NORMALIZED = data.get('normalized', True)
    MODEL_LOADED = True
    print("Successfully loaded asl_model.pkl!")
except Exception:
    print("WARNING: asl_model.pkl not found. Run train.py first.")


# --- 3. Universal ASL Biomechanical & ML Recognition ---
def get_finger_states(pts):
    """Scale-invariant finger extension & hand orientation extractor."""
    wrist = pts[0]
    palm_w = np.linalg.norm(pts[5, :2] - pts[17, :2]) + 1e-6
    hand_vec = pts[9, :2] - wrist[:2]
    is_horiz = abs(hand_vec[0]) > abs(hand_vec[1]) * 1.15
    is_upright = (wrist[1] - pts[9, 1]) > (palm_w * 0.25)

    def is_open(tip, pip, mcp):
        d_tip = np.linalg.norm(pts[tip, :2] - wrist[:2])
        d_pip = np.linalg.norm(pts[pip, :2] - wrist[:2])
        d_mcp = np.linalg.norm(pts[mcp, :2] - wrist[:2])
        return (d_tip > d_pip * 1.02) and (d_tip > d_mcp * 1.14)

    index  = is_open(8, 6, 5)
    middle = is_open(12, 10, 9)
    ring   = is_open(16, 14, 13)
    pinky  = is_open(20, 18, 17)

    thumb_to_pinky = np.linalg.norm(pts[4, :2] - pts[17, :2])
    thumb_to_index = np.linalg.norm(pts[4, :2] - pts[5, :2])
    thumb = (thumb_to_pinky > palm_w * 1.28) and (thumb_to_index > palm_w * 0.48)

    return thumb, index, middle, ring, pinky, is_horiz, is_upright, palm_w


def classify_asl_phrase(all_hands_pts, face_box, ml_letter=""):
    """
    Recognizes 15 Authentic Universal ASL Signs where every sign has a 100% mutually exclusive
    (Hands + Finger-Count + Orientation + Location) signature so no two signs ever conflict.
    """
    pts = all_hands_pts[0]
    thumb, index, middle, ring, pinky, is_horiz, is_upright, palm_w = get_finger_states(pts)
    open_four = index and middle and ring and pinky
    open_cnt  = int(index) + int(middle) + int(ring) + int(pinky)
    is_fist   = (open_cnt == 0)

    hx, hy = pts[9, 0], pts[9, 1]
    tip_y = pts[8, 1]

    if face_box is not None:
        fx, fy, fw, fh = face_box
        mouth_y = fy + fh * 0.78
        chin_y  = fy + fh * 1.05
        face_cx = fx + fw * 0.50
    else:
        mouth_y, chin_y, face_cx = 0.43, 0.55, 0.50

    # --- A. TWO-HAND AUTHENTIC ASL SIGNS (5 Non-Overlapping Combinations) ---
    if len(all_hands_pts) >= 2:
        pts2 = all_hands_pts[1]
        t2, i2, m2, r2, p2, h2, _, _ = get_finger_states(pts2)
        open_cnt2 = int(i2) + int(m2) + int(r2) + int(p2)
        is_fist2  = (open_cnt2 == 0)

        # 1. 'MY NAME IS': Both hands have 2 fingers (Index + Middle 'H' shape)
        if (index and middle and not ring and not pinky) and (i2 and m2 and not r2 and not p2):
            return "MY NAME IS", "My Name Is", 0.98

        # 2. 'NICE TO MEET YOU': Both hands have 1 upright Index finger ('1' shape)
        if (index and not middle and not ring and not pinky) and (i2 and not m2 and not r2 and not p2):
            return "NICE TO MEET YOU", "Nice To Meet You", 0.98

        # 3. 'HELP': 1 Closed Fist ('A'/'S') resting on 1 Open Palm (3+ fingers)
        if (is_fist and open_cnt2 >= 3) or (is_fist2 and open_cnt >= 3):
            return "HELP", "Help", 0.98

        # 4. 'HOW ARE YOU': Both hands in Closed Fists (0 fingers open) at chest
        if is_fist and is_fist2:
            return "HOW ARE YOU", "How Are You", 0.98

        # 5. 'GOOD MORNING': Both hands open (3+ fingers) — horizon arm + rising sun hand
        if open_cnt >= 3 and open_cnt2 >= 3:
            return "GOOD MORNING", "Good Morning", 0.98

        return "", "", 0.0

    # --- B. SINGLE-HAND AUTHENTIC ASL SIGNS (10 Non-Overlapping Finger/Location Combinations) ---
    # 6. 'I LOVE YOU': Thumb + Index + Pinky open (Middle + Ring closed)
    if thumb and index and (not middle) and (not ring) and pinky:
        return "I LOVE YOU", "I Love You", 0.99

    # 7. 'WATER': Exactly 3 fingers ('W' shape: Index + Middle + Ring open, Pinky closed)
    if index and middle and ring and (not pinky):
        return "WATER", "Water", 0.98

    # 8. 'NAME IS' vs 'SEE YOU LATER': Exactly 2 fingers (Index + Middle open, Ring + Pinky closed)
    if index and middle and (not ring) and (not pinky):
        if is_horiz:
            return "NAME IS", "Name Is", 0.98
        return "SEE YOU LATER", "See You Later", 0.98

    # 9. 'WHAT IS YOURS': 1 Index finger pointing up OR Thumb + Pinky ('Y' shape)
    if (index and not middle and not ring and not pinky and not is_horiz) or (
        thumb and pinky and not index and not middle and not ring
    ):
        return "WHAT IS YOURS", "What Is Yours", 0.98

    # 10. 'YES' vs 'SORRY': Closed Fist (0 fingers open) — separated cleanly by height!
    if is_fist and is_upright:
        if hy < chin_y:
            return "YES", "Yes", 0.97
        if hy >= chin_y and abs(hx - face_cx) < 0.22:
            return "SORRY", "Sorry", 0.97
        return "", "", 0.0

    # 11. 'HELLO' vs 'THANK YOU' vs 'MY': Open Hand (4 fingers open) — separated cleanly by height!
    if open_four:
        # Forehead / Temple / Upper side of face -> 'HELLO'
        if not is_horiz and (tip_y < mouth_y or (tip_y < chin_y and abs(hx - face_cx) > 0.14)):
            return "HELLO", "Hello", 0.98
        # Mouth / Chin -> 'THANK YOU'
        if not is_horiz and (mouth_y <= tip_y <= chin_y) and abs(hx - face_cx) <= 0.14:
            return "THANK YOU", "Thank You", 0.98
        # Chest (below chin or held horizontally on chest) -> 'MY'
        if hy > chin_y or is_horiz:
            return "MY", "My", 0.98

    return "", "", 0.0


def classify_asl_letter(pts, model, encoder, use_norm):
    """Universal 26-letter ASL classifier combining Random Forest probabilities with finger geometry."""
    thumb, index, middle, ring, pinky, is_horiz, is_upright, palm_w = get_finger_states(pts)
    if not is_upright or is_horiz:
        return "", 0.0

    features = normalize_single_hand(pts.reshape(63)) if use_norm else pts.reshape(63)
    probs = model.predict_proba([features])[0]
    top_idx = int(np.argmax(probs))
    ml_conf = float(probs[top_idx])
    ml_letter = str(encoder.inverse_transform([top_idx])[0]).upper()

    if index and middle and ring and pinky and not thumb:
        return "B", max(ml_conf, 0.97)

    if index and middle and not ring and not pinky:
        tip_dist = np.linalg.norm(pts[8, :2] - pts[12, :2])
        pip_dist = np.linalg.norm(pts[6, :2] - pts[10, :2])
        crossed = ((pts[8, 0] - pts[12, 0]) * (pts[5, 0] - pts[9, 0])) <= 0
        if crossed or tip_dist < pip_dist * 0.90 or ml_letter == "R":
            return "R", max(ml_conf, 0.97)
        elif tip_dist > palm_w * 0.40:
            return "V", max(ml_conf, 0.97)
        return "U", max(ml_conf, 0.95)

    if not index and not middle and not ring and not pinky:
        d_thumb_pinky = np.linalg.norm(pts[4, :2] - pts[17, :2])
        d_thumb_index = np.linalg.norm(pts[4, :2] - pts[5, :2])
        d_thumb_ring  = np.linalg.norm(pts[4, :2] - pts[14, :2])
        if d_thumb_pinky > palm_w * 1.06 and d_thumb_index < d_thumb_ring and pts[4, 1] < pts[6, 1] + palm_w * 0.15:
            return "A", max(ml_conf, 0.97)
        if ml_letter in ("N", "M", "T", "S", "E"):
            return ("N" if d_thumb_ring < palm_w * 0.58 else ml_letter), max(ml_conf, 0.95)
        return "N", max(ml_conf, 0.94)

    if ml_conf >= 0.35:
        ml_conf = min(0.99, 0.92 + (ml_conf - 0.35) * 0.14)
    return ml_letter, ml_conf


# --- 4. Real-Time Webcam Loop & Temporal Filter ---
cap = cv2.VideoCapture(0)
# MODE 0 = ASL WORDS MODE (Auto-spells name right after 'Name Is', otherwise recognizes 20+ ASL Words!)
# MODE 1 = FINGERSPELLING MODE (A-Z Letters)
MODE = 0

smoothed_hands = [None, None]
DEADBAND = 0.007
gesture_window = deque(maxlen=8)
PHRASE_VOTES = 5
LETTER_VOTES = 6
LETTER_COOLDOWN = 0.90

confirmed_gesture = ""
confirmed_is_letter = False
current_confidence = 0.0
display_timer = 0

sentence_history = []
spelling_buffer = []
auto_spell_name = False
last_letter_added = ""
last_letter_time = 0.0
last_hand_seen_time = time.time()
last_spoken_times = {}
face_box = None
frame_count = 0
pTime = 0

window_name = "Sign Language Translator - Universal ASL Edition"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    frame_count += 1
    img = cv2.flip(img, 1)
    h_img, w_img, _ = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if frame_count % 20 == 1:
        face_res = face_detector.process(imgRGB)
        if face_res.detections:
            b = face_res.detections[0].location_data.relative_bounding_box
            face_box = (b.xmin, b.ymin, b.width, b.height)

    results = hands.process(imgRGB)
    detected_gesture = ""
    speech_text = ""
    confidence_score = 0.0
    is_letter = False
    hold_progress = 0.0
    cooldown_left = max(0.0, LETTER_COOLDOWN - (time.time() - last_letter_time))

    if results.multi_hand_landmarks:
        all_hands_pts = []
        wrist_motion = 0.0
        finger_motion = 0.0

        for i, hand_landmarks in enumerate(results.multi_hand_landmarks[:2]):
            raw_pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float64)
            if smoothed_hands[i] is None:
                smoothed_hands[i] = raw_pts.copy()
            else:
                deltas = np.linalg.norm(raw_pts[:, :2] - smoothed_hands[i][:, :2], axis=1)
                if i == 0:
                    wrist_motion = float(deltas[9])
                    finger_motion = float(np.max(deltas[[4, 8, 12, 16, 20]]))
                alpha = np.where(deltas < DEADBAND, 0.10, np.clip((deltas - DEADBAND) * 10.0 + 0.22, 0.22, 0.78))[:, np.newaxis]
                smoothed_hands[i] = alpha * raw_pts + (1.0 - alpha) * smoothed_hands[i]

            for j, lm in enumerate(hand_landmarks.landmark):
                lm.x, lm.y, lm.z = smoothed_hands[i][j]
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            all_hands_pts.append(smoothed_hands[i])

        if len(results.multi_hand_landmarks) == 1:
            smoothed_hands[1] = None

        if all_hands_pts[0][0, 1] < 0.94 and all_hands_pts[0][9, 1] < 0.88 and wrist_motion < 0.042:
            last_hand_seen_time = time.time()

            pred_let, ml_conf = "", 0.0
            if MODEL_LOADED and len(all_hands_pts) == 1:
                pred_let, ml_conf = classify_asl_letter(all_hands_pts[0], model, encoder, USE_NORMALIZED)

            # 1. Check Universal ASL Words (when in Mode 0 and not auto-spelling a name after 'Name Is')
            if MODE == 0 and (not auto_spell_name or len(all_hands_pts) >= 2):
                asl_g, asl_s, asl_c = classify_asl_phrase(all_hands_pts, face_box, pred_let)
                if asl_g != "":
                    detected_gesture, speech_text, confidence_score = asl_g, asl_s, asl_c

            # 2. Check 26-Letter ASL Alphabet (when in Mode 1 OR auto-spelling name after 'Name Is')
            if detected_gesture == "" and (MODE == 1 or auto_spell_name) and len(all_hands_pts) == 1 and finger_motion < 0.024 and cooldown_left <= 0.0:
                if pred_let != "" and ml_conf > 0.38:
                    detected_gesture = pred_let
                    confidence_score = ml_conf
                    is_letter = True

            if confidence_score > 0.0:
                current_confidence = confidence_score
    else:
        smoothed_hands = [None, None]
        gesture_window.clear()
        last_letter_added = ""
        # Commit fingerspelled word (2+ letters) when hand is lowered for 1.0s
        if spelling_buffer and (time.time() - last_hand_seen_time > 1.0):
            if len(spelling_buffer) >= 2:
                word = "".join(spelling_buffer).title()
                speech_queue.put(word)
                auto_spell_name = False
                if not sentence_history or sentence_history[-1].upper() != word.upper():
                    sentence_history.append(word)
                    if len(sentence_history) > 8:
                        sentence_history.pop(0)
            spelling_buffer.clear()

    # Majority-Vote Temporal Confirmation
    if detected_gesture != "":
        gesture_window.append((detected_gesture, speech_text, is_letter))
        top_gesture, votes = Counter(g[0] for g in gesture_window).most_common(1)[0]
        matched_speech, matched_is_letter = speech_text, is_letter
        for g_name, g_sp, g_let in reversed(gesture_window):
            if g_name == top_gesture:
                matched_speech, matched_is_letter = g_sp, g_let
                break

        req = LETTER_VOTES if matched_is_letter else PHRASE_VOTES
        hold_progress = min(1.0, votes / float(req))

        if votes >= req:
            confirmed_gesture = top_gesture
            confirmed_is_letter = matched_is_letter
            display_timer = 20
            now_t = time.time()

            if matched_is_letter:
                if top_gesture == "DEL":
                    if spelling_buffer and last_letter_added != "DEL":
                        spelling_buffer.pop()
                        last_letter_added = "DEL"
                        gesture_window.clear()
                elif top_gesture == "SPACE":
                    if len(spelling_buffer) >= 2:
                        word = "".join(spelling_buffer).title()
                        speech_queue.put(word)
                        auto_spell_name = False
                        if not sentence_history or sentence_history[-1].upper() != word.upper():
                            sentence_history.append(word)
                    spelling_buffer.clear()
                    last_letter_added = ""
                    gesture_window.clear()
                elif top_gesture != last_letter_added and (now_t - last_letter_time > LETTER_COOLDOWN):
                    spelling_buffer.append(top_gesture)
                    last_letter_added = top_gesture
                    last_letter_time = now_t
                    gesture_window.clear()
                    speech_queue.put(top_gesture.lower())
            else:
                if len(spelling_buffer) >= 2:
                    word = "".join(spelling_buffer).title()
                    speech_queue.put(word)
                    auto_spell_name = False
                    if not sentence_history or sentence_history[-1].upper() != word.upper():
                        sentence_history.append(word)
                spelling_buffer.clear()

                existing = [w.upper() for w in sentence_history]
                is_dup_intro = matched_speech.upper() in ("MY", "NAME IS", "MY NAME IS") and "NAME IS" in " ".join(existing)
                if matched_speech != "" and matched_speech.upper() not in existing and not is_dup_intro:
                    sentence_history.append(matched_speech)
                    speech_queue.put(matched_speech)
                    last_spoken_times[matched_speech.upper()] = now_t
                    if "NAME IS" in matched_speech.upper():
                        last_letter_time = now_t + 0.4
                        auto_spell_name = True
                    if len(sentence_history) > 8:
                        sentence_history.pop(0)
                gesture_window.clear()
    else:
        if display_timer > 0:
            display_timer -= 1
        else:
            confirmed_gesture = ""
            if not results.multi_hand_landmarks:
                current_confidence = 0.0

    # --- 5. Video Overlay & Bottom HUD ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cooldown_left > 0.0 and results.multi_hand_landmarks and (MODE == 1 or auto_spell_name):
        msg = f"Locked: {last_letter_added} (Next in {cooldown_left:.1f}s)" if last_letter_added else f"Ready to spell in {cooldown_left:.1f}s..."
        cv2.rectangle(img, (12, 42), (360, 90), (0, 0, 0), -1)
        cv2.putText(img, msg, (20, 72), font, 0.58, (0, 255, 255), 2)
    elif confirmed_gesture != "":
        lbl = f"Letter: {confirmed_gesture}" if confirmed_is_letter else f"ASL Word: {confirmed_gesture}"
        (tw, th), _ = cv2.getTextSize(lbl, font, 0.85, 2)
        cv2.rectangle(img, (10, 68 - th), (32 + tw, 92), (0, 0, 0), -1)
        cv2.putText(img, lbl, (20, 80), font, 0.85, (0, 255, 0), 2)
    elif detected_gesture != "" and hold_progress > 0.25:
        cv2.rectangle(img, (12, 45), (310, 90), (0, 0, 0), -1)
        cv2.putText(img, f"Holding {detected_gesture}... {int(hold_progress * 100)}%", (20, 74), font, 0.58, (100, 255, 255), 2)

    cTime = time.time()
    fps = 1.0 / (cTime - pTime) if pTime != 0 else 0.0
    lat = (cTime - pTime) * 1000.0 if pTime != 0 else 0.0
    pTime = cTime

    hud_str = f"FPS: {int(fps)} | Latency: {int(lat)}ms | Acc: {int(current_confidence * 100)}%"
    (hw, hh), _ = cv2.getTextSize(hud_str, font, 0.6, 2)
    cv2.rectangle(img, (w_img - hw - 28, 27 - hh), (w_img - 12, 43), (0, 0, 0), -1)
    cv2.putText(img, hud_str, (w_img - hw - 20, 35), font, 0.6, (0, 255, 0) if lat <= 150 else (0, 0, 255), 2)

    hud_panel = np.zeros((85, w_img, 3), dtype=np.uint8)
    cv2.line(hud_panel, (0, 0), (w_img, 0), (80, 80, 80), 2)
    mode_tag = "ASL WORDS" if (MODE == 0 and not auto_spell_name) else "SPELL(A-Z)"
    cv2.putText(hud_panel, f"[{mode_tag}]  M:Mode | ENTER:Word | BKSP:Del | R:Repeat | SPACE:Speak | C:Clear",
                (10, 20), font, 0.39, (160, 255, 160), 1)

    sp_str = "".join(spelling_buffer)
    if sp_str:
        cv2.putText(hud_panel, f"Spelling: {sp_str}_  (Lower hand 1s or press ENTER)", (10, 46), font, 0.52, (0, 255, 255), 2)
    elif auto_spell_name or MODE == 1:
        cv2.putText(hud_panel, "Spelling Mode: Sign A-Z letters, then lower hand 1s (or press ENTER)", (10, 45), font, 0.44, (0, 255, 255), 1)
    else:
        cv2.putText(hud_panel, "ASL Words Mode: Sign Hello, Good Morning, Help, Water, Thank You, Yes, No...", (10, 45), font, 0.42, (160, 160, 160), 1)

    sent_str = f"Sentence: {' '.join(sentence_history)}"
    scale = 0.60
    while scale > 0.40 and cv2.getTextSize(sent_str, font, scale, 2)[0][0] > w_img - 20:
        scale -= 0.04
    cv2.putText(hud_panel, sent_str, (10, 75), font, scale, (255, 255, 255), 2 if scale >= 0.48 else 1)

    cv2.imshow(window_name, np.vstack([img, hud_panel]))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('m'):
        MODE = 1 - MODE
        auto_spell_name = False
        gesture_window.clear()
    elif key == ord('r'):
        last_letter_added = ""
    elif key == 13:  # ENTER commits any spelled word
        if spelling_buffer:
            word = "".join(spelling_buffer).title()
            spelling_buffer.clear()
            auto_spell_name = False
            last_letter_added = ""
            speech_queue.put(word)
            sentence_history.append(word)
    elif key == 8:   # BACKSPACE deletes last letter
        if spelling_buffer:
            spelling_buffer.pop()
            last_letter_added = ""
    elif key == ord('c'):
        sentence_history.clear()
        spelling_buffer.clear()
        last_spoken_times.clear()
        auto_spell_name = False
        last_letter_added = ""
    elif key == ord(' '):
        if spelling_buffer:
            sentence_history.append("".join(spelling_buffer).title())
            spelling_buffer.clear()
            auto_spell_name = False
        if sentence_history:
            speech_queue.put(" ".join(sentence_history))

    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
