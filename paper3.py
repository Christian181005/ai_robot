import cv2
import numpy as np
from picamera2 import Picamera2
import time

def erkenne_schwarzen_balken(frame):
    # 1) Graustufen + invertiertes Threshold (nur sehr dunkle Pixel)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # 2) Morphologisches Closing, um Löcher im Block zu füllen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 3) Konturen finden
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Benutzerdefinierter Flächenfilter (wie gewünscht beibehalten)
        if area < 800 and area > 2000:
            continue

        # 4) Polygon-Approximation: nur Vierecke weiterverarbeiten
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) != 4:
            continue

        # 5) gedrehtes Rechteck und Aspect-Ratio (ca. 1:3)
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        ar = max(w, h) / min(w, h)
        if not (2.0 < ar < 5.0):
            continue

        # 6) Solidity-Filter: sicherstellen, dass das Rechteck voll gefüllt ist
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < 0.9:
            continue

        # 7) erkannte Kontur einzeichnen
        cv2.drawContours(frame, [box], -1, (0, 255, 0), 3)
        cx, cy = int(rect[0][0]), int(rect[0][1])
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{cx},{cy}", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

def main():
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()
    time.sleep(1)

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output = erkenne_schwarzen_balken(frame)
            cv2.imshow("Erkennung schwarzer Balken", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
