import cv2
import mediapipe as mp
import pyttsx3
import threading
import math
import time

# --- AUDIO DEADLOCK FIX: Micro-Thread per word ---
def speak_word(text):
    def run_tts():
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    
    # Fire and forget: spawns a fresh audio thread that dies as soon as it finishes speaking
    threading.Thread(target=run_tts, daemon=True).start()

def get_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

print("Initializing camera... Please wait a moment for the window to pop up.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("CRITICAL ERROR: Windows is blocking the webcam.")

pending_gesture = ""
hold_frames = 0
REQUIRED_FRAMES = 12  

confirmed_gesture = ""
confirmed_speech = ""
display_timer = 0

already_spoken = False
current_spoken_word = ""
last_added_word = ""
sentence_history = []
pTime = 0

window_name = "Sign Language Translator MVP"
cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    detected_gesture = ""
    speech_text = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            fingers = []
            wrist = hand_landmarks.landmark[0]
            pinky_base = hand_landmarks.landmark[17]
            
            thumb_tip = hand_landmarks.landmark[4]
            thumb_joint = hand_landmarks.landmark[3]
            
            if get_distance(thumb_tip, pinky_base) > get_distance(thumb_joint, pinky_base):
                fingers.append(1)
            else:
                fingers.append(0)

            tip_ids = [8, 12, 16, 20]
            pip_ids = [6, 10, 14, 18]

            for tip, pip in zip(tip_ids, pip_ids):
                if get_distance(hand_landmarks.landmark[tip], wrist) > get_distance(hand_landmarks.landmark[pip], wrist):
                    fingers.append(1)
                else:
                    fingers.append(0)

            if fingers == [1, 1, 1, 1, 1]:
                detected_gesture = "HELLO (OPEN PALM)"
                speech_text = "Hello"
            elif fingers == [0, 1, 1, 1, 1]:
                detected_gesture = "PLEASE"
                speech_text = "Please"
            elif fingers == [1, 0, 0, 0, 0]:
                if thumb_tip.y < pinky_base.y:
                    detected_gesture = "GOOD / YES"
                    speech_text = "Good"
                else:
                    detected_gesture = "BAD / NO"
                    speech_text = "Bad"
            elif fingers == [0, 1, 0, 0, 0]:
                detected_gesture = "YOU (POINTING)"
                speech_text = "You"
            elif fingers == [0, 0, 0, 0, 1]:
                detected_gesture = "ME / I"
                speech_text = "I"
            elif fingers == [0, 1, 1, 0, 0]:
                detected_gesture = "PEACE / V"
                speech_text = "Peace"
            elif fingers == [1, 1, 0, 0, 0]:
                detected_gesture = "LATER / L"
                speech_text = "Later"
            elif fingers == [1, 1, 0, 0, 1]:
                detected_gesture = "I LOVE YOU"
                speech_text = "I Love You"
            elif fingers == [1, 0, 0, 0, 1]:
                detected_gesture = "CALL ME / Y"
                speech_text = "Call Me"
            elif fingers == [0, 0, 1, 1, 1]:
                detected_gesture = "OKAY / F"
                speech_text = "Okay"
            elif fingers == [0, 1, 1, 1, 0]:
                detected_gesture = "WATER / W"
                speech_text = "Water"
            elif fingers == [1, 1, 1, 0, 0]:
                detected_gesture = "THREE"
                speech_text = "Three"
            elif fingers == [0, 1, 0, 0, 1]:
                detected_gesture = "ROCK ON"
                speech_text = "Rock On"
            elif fingers == [0, 0, 0, 0, 0]:
                detected_gesture = "WAITING... (FIST)"
                speech_text = ""  

    if detected_gesture != "":
        if detected_gesture == pending_gesture:
            hold_frames += 1
        else:
            pending_gesture = detected_gesture
            hold_frames = 1 
            
        if hold_frames >= REQUIRED_FRAMES:
            confirmed_gesture = detected_gesture
            confirmed_speech = speech_text
            display_timer = 30 
    else:
        pending_gesture = ""
        hold_frames = 0
        if display_timer > 0:
            display_timer -= 1
        else:
            confirmed_gesture = ""
            confirmed_speech = ""
            already_spoken = False

    if confirmed_gesture != "":
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 3
        
        text_size, _ = cv2.getTextSize(confirmed_gesture, font, font_scale, font_thickness)
        text_w, text_h = text_size
        
        x, y = 50, 100
        padding = 15
        
        overlay = img.copy()
        cv2.rectangle(overlay, (x - padding, y - text_h - padding), (x + text_w + padding, y + padding), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        cv2.putText(img, confirmed_gesture, (x, y), font, font_scale, (0, 255, 0), font_thickness)
        
        if confirmed_speech != "":
            if not already_spoken or current_spoken_word != confirmed_speech:
                # --- NEW MICRO-THREAD CALL ---
                speak_word(confirmed_speech)
                already_spoken = True
                current_spoken_word = confirmed_speech
                
                if confirmed_speech != last_added_word:
                    sentence_history.append(confirmed_speech)
                    last_added_word = confirmed_speech
                    if len(sentence_history) > 4: 
                        sentence_history.pop(0)
        
        if confirmed_gesture == "WAITING... (FIST)":
            last_added_word = ""
            current_spoken_word = "" 
            already_spoken = False

    h_img, w_img, _ = img.shape
    cv2.rectangle(img, (0, h_img - 60), (w_img, h_img), (0, 0, 0), -1)
    
    sentence_str = " ".join(sentence_history)
    ticker_text = f"Sentence: {sentence_str}"
    cv2.putText(img, ticker_text, (20, h_img - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    latency_ms = (cTime - pTime) * 1000 if pTime != 0 else 0
    pTime = cTime

    fps_text = f"FPS: {int(fps)} | Latency: {int(latency_ms)}ms"
    hud_font = cv2.FONT_HERSHEY_SIMPLEX
    hud_scale = 0.7
    hud_thick = 2
    
    hud_size, _ = cv2.getTextSize(fps_text, hud_font, hud_scale, hud_thick)
    hud_w, hud_h = hud_size
    
    hud_x = img.shape[1] - hud_w - 30
    hud_y = 40
    hud_pad = 10
    
    hud_overlay = img.copy()
    cv2.rectangle(hud_overlay, (hud_x - hud_pad, hud_y - hud_h - hud_pad), 
                  (hud_x + hud_w + hud_pad, hud_y + hud_pad), (0, 0, 0), -1)
    cv2.addWeighted(hud_overlay, 0.6, img, 0.4, 0, img)
    
    hud_color = (0, 255, 0) if latency_ms <= 150 else (0, 0, 255)
    cv2.putText(img, fps_text, (hud_x, hud_y), hud_font, hud_scale, hud_color, hud_thick)

    cv2.imshow(window_name, img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
