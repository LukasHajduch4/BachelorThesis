import cv2
import torch
import time
from ultralytics import YOLOWorld

# Inicializuj model bez nastavených tried = open-vocabulary
device = "cpu"
model = YOLOWorld("yolov8m-worldv2").to(device)

# Kamera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera sa nedá otvoriť")
    exit()

print("✅ YOLO-World spustený v open-vocabulary režime. Stlač Q pre ukončenie.")
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Snímok sa nepodarilo získať")
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb)[0]  # žiadne text=, žiadne set_classes()

    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if conf < 0.2:
            continue
        x1, y1, x2, y2 = map(int, box)
        label = model.names[int(cls)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("YOLO-World – Open Vocabulary", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
