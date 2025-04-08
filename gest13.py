#!/usr/bin/env python3
import gpiod
import threading
import time
import cv2
import mediapipe as mp
from picamera2 import Picamera2
import math

# ========================
# Servo-PWM-Konfiguration (libgpiod, softwarebasiert)
# ========================
CHIP = "/dev/gpiochip0"       # Standardmäßig gpiochip0
LINE_OFFSET = 18              # Funktionierender GPIO-Offset laut Testprogramm
PWM_PERIOD = 0.02             # 20 ms Periode (50 Hz)
MIN_PULSE = 0.0001             # 1 ms Pulsbreite für 0° (anpassen, falls nötig)
MAX_PULSE = 0.004             # 2 ms Pulsbreite für 180° (anpassen, falls nötig)

# Globaler Servowinkel in Grad (Initialwert 90° = Mittelstellung)
servo_angle = 90  
angle_lock = threading.Lock()  # Schutz für servo_angle

def angle_to_pulse(angle):
    """Wandelt einen Winkel (0° bis 180°) in eine Pulsbreite (in Sekunden) um."""
    angle = max(0, min(angle, 180))
    pulse = MIN_PULSE + (MAX_PULSE - MIN_PULSE) * (angle / 180.0)
    return pulse

def pwm_cycle(line, pulse_width):
    """Führt einen PWM-Zyklus aus: Schaltet den Ausgang für pulse_width Sekunden HIGH und den Rest der Periode LOW."""
    line.set_value(1)
    time.sleep(pulse_width)
    line.set_value(0)
    time.sleep(PWM_PERIOD - pulse_width)

def servo_pwm_thread(line):
    """Dieser Thread gibt kontinuierlich PWM-Zyklen ab, basierend auf dem aktuell gewünschten Winkel."""
    global servo_angle
    while True:
        # Lese den aktuellen Winkel thread-sicher
        with angle_lock:
            current_angle = servo_angle
        pulse_width = angle_to_pulse(current_angle)
        pwm_cycle(line, pulse_width)

# ========================
# Medienverarbeitung (Picamera2 und MediaPipe)
# ========================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

picam2 = Picamera2()
cam_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(cam_config)
picam2.start()

def calculate_finger_length(hand_landmarks, frame_shape):
    """Berechnet die Euklidische Distanz zwischen MCP- und TIP-Landmark des Zeigefingers."""
    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    
    h, w = frame_shape[:2]
    mcp_pos = (int(mcp.x * w), int(mcp.y * h))
    tip_pos = (int(tip.x * w), int(tip.y * h))
    
    dx = tip_pos[0] - mcp_pos[0]
    dy = tip_pos[1] - mcp_pos[1]
    length = math.sqrt(dx**2 + dy**2)
    return length, mcp_pos, tip_pos

def calculate_hand_height(hand_landmarks, frame_shape):
    """Berechnet die Höhe der Hand anhand aller Landmark-Punkte."""
    h, w = frame_shape[:2]
    ys = [landmark.y * h for landmark in hand_landmarks.landmark]
    return max(ys) - min(ys)

def map_value(x, in_min, in_max, out_min, out_max):
    """Lineare Interpolation mit Clamping: Wert x aus dem Bereich [in_min, in_max] wird auf [out_min, out_max] abgebildet."""
    if x < in_min:
        x = in_min
    if x > in_max:
        x = in_max
    return out_min + (x - in_min) * (out_max - out_min) / (in_max - in_min)

# ========================
# Hauptprogramm
# ========================
def main():
    global servo_angle
    # Libgpiod: Öffne den GPIO-Chip und fordere den gewünschten Pin als Ausgang an
    chip = gpiod.Chip(CHIP)
    line = chip.get_line(LINE_OFFSET)
    line.request(consumer="servo-test", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
    
    # Starte den PWM-Thread (Daemon, läuft im Hintergrund)
    pwm_thread = threading.Thread(target=servo_pwm_thread, args=(line,), daemon=True)
    pwm_thread.start()
    
    print("Starte Servo- und Handtracking-Steuerung. Drücke 'q' im Fenster zum Beenden.")
    
    # MediaPipe: Hands-Erkennung
    with mp_hands.Hands(
        static_image_mode=False,
        min_detection_confidence=0.8,
        max_num_hands=1
    ) as hands:
        while True:
            # Kamerabild erfassen und verarbeiten
            frame = picam2.capture_array()
            # PiCamera2 liefert meist im RGB-Format; cv2 erwartet BGR, also zwei Umwandlungen:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Zeichne Handlandmarks
                    mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Berechne Fingerlänge und Handhöhe
                    finger_length, mcp, tip = calculate_finger_length(hand_landmarks, frame_bgr.shape)
                    hand_height = calculate_hand_height(hand_landmarks, frame_bgr.shape)
                    
                    if hand_height != 0:
                        relative_length = finger_length / hand_height
                    else:
                        relative_length = 0
                    
                    cv2.arrowedLine(frame_bgr, mcp, tip, (0, 255, 0), 3)
                    cv2.putText(frame_bgr, f"Relativ: {relative_length:.2f}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Mappen des relativen Werts (angenommener Bereich 0.2 bis 1.0)
                    # auf einen Winkel von 10° bis 170°
                    new_angle = map_value(relative_length, 0.2, 1.0, 10, 170)
                    # Aktualisiere den globalen Winkel thread-sicher
                    with angle_lock:
                        servo_angle = new_angle
            
            cv2.imshow("Handtracking", frame_bgr)
            if cv2.waitKey(1) == ord('q'):
                break
    
    # Aufräumen
    picam2.stop()
    cv2.destroyAllWindows()
    line.set_value(0)
    line.release()

if __name__ == '__main__':
    main()
