import os
# os.environ["GPIOZERO_PIN_FACTORY"] = "gpiod"  # Falls nötig, damit gpiozero libgpiod verwendet
from gpiozero import AngularServo
import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2
import math

# Servo initialisieren und in Mittelstellung setzen
servo = AngularServo(18, min_angle=0, max_angle=180,
                     min_pulse_width=0.0005, max_pulse_width=0.0025)
servo.angle = 90
print("Servo in Mittelstellung (90°)")
time.sleep(1)

# MediaPipe initialisieren
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Pi-Kamera konfigurieren
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()

def calculate_finger_length(hand_landmarks, frame_shape):
    # Landmark-Punkte des Zeigefingers (MCP und TIP)
    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    # Umrechnung in Pixel-Koordinaten
    h, w = frame_shape[:2]
    mcp_pos = (int(mcp.x * w), int(mcp.y * h))
    tip_pos = (int(tip.x * w), int(tip.y * h))
    
    # Berechnung der euklidischen Distanz
    dx = tip_pos[0] - mcp_pos[0]
    dy = tip_pos[1] - mcp_pos[1]
    length = math.sqrt(dx**2 + dy**2)
    return length, mcp_pos, tip_pos

try:
    with mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.7,
        max_num_hands=1
    ) as hands:
        while True:
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Zeichne die Handlandmarks
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    
                    # Berechne die Länge des Zeigefingers
                    length, mcp, tip = calculate_finger_length(hand_landmarks, frame_bgr.shape)
                    
                    # Visualisierung: grüner Pfeil und Textanzeige
                    cv2.arrowedLine(frame_bgr, mcp, tip, (0, 255, 0), 3)
                    cv2.putText(
                        frame_bgr, 
                        f"Fingerauslenkung: {length:.1f}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    
                    # Steuerung des Servos: < 40 Pixel -> 0°, >= 40 Pixel -> 180°
                    if length < 40:
                        servo.angle = 0
                    else:
                        servo.angle = 180
            
            cv2.imshow("Index Finger Tracking", frame_bgr)
            if cv2.waitKey(1) == ord('q'):
                break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
