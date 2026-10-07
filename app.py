import cv2
import mediapipe as mp
import pyttsx3
import threading
import queue

speech_queue = queue.Queue()

def tts_worker():
    """A dedicated background thread that processes speech safely one at a time."""
    engine = pyttsx3.init()
    while True:
        text = speech_queue.get()
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()


threading.Thread(target=tts_worker, daemon=True).start()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(r"C:\Users\Arnav Thombre\OneDrive\Desktop\Sign-Language-MVP\my_sign_video.mp4")

already_spoken = False
current_spoken_word = ""

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            fingers = []
            
            if hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y and hand_landmarks.landmark[4].y < hand_landmarks.landmark[5].y:
                fingers.append(1)
            else:
                fingers.append(0)

            tip_ids = [8, 12, 16, 20]
            base_ids = [6, 10, 14, 18] 

            for tip, base in zip(tip_ids, base_ids):
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[base].y:
                    fingers.append(1) 
                else:
                    fingers.append(0) 

            gesture_text = ""
            speech_text = ""

            if fingers == [1, 1, 1, 1, 1] or fingers == [0, 1, 1, 1, 1]:
                gesture_text = "HELLO (OPEN PALM)"
                speech_text = "Hello"
            elif fingers == [1, 0, 0, 0, 0]:
                gesture_text = "YES (THUMBS UP)"
                speech_text = "Yes"
            elif fingers == [0, 1, 1, 0, 0] or fingers == [1, 1, 1, 0, 0]:
                gesture_text = "PEACE (V-SIGN)"
                speech_text = "Peace"
            elif fingers == [0, 0, 0, 0, 0]: 
                gesture_text = "FIST (CLOSED)"
                speech_text = "Fist"

            if gesture_text != "":
                cv2.putText(img, gesture_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                if not already_spoken or current_spoken_word != speech_text:
                    
                    speech_queue.put(speech_text)
                    
                    already_spoken = True
                    current_spoken_word = speech_text
            else:
                already_spoken = False
                current_spoken_word = ""

    cv2.imshow("Sign Language Translator MVP", img)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
