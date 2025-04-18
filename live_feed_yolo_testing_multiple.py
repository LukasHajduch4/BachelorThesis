import cv2
import time
from ultralytics import YOLO

# 📦 Modely na prepínanie
model_paths = {
    ord("1"): "yolov8n.pt",
    ord("2"): "yolov8s.pt",
    ord("3"): "yolo11n.pt",
    ord("4"): "yolo11s.pt",
}

current_key = ord("1")
current_model_name = model_paths[current_key]
model = YOLO(current_model_name)
model.to("cpu")
model.eval()
print(f"✅ Spustený model: {current_model_name}")

# 🎥 Kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kamera sa nedá načítať.")
        break

    # 🎯 Zmenšenie pre výkon
    resized = cv2.resize(frame, (640, 480))

    # 🔍 Detekcia
    results = model(resized, verbose=False)
    annotated = results[0].plot()

    # 🕒 Výpočet FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # ✏️ Vykresli FPS a názov modelu
    cv2.putText(annotated, f"Model: {current_model_name}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(annotated, f"FPS: {fps:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # 🖼️ Zobraz výstup
    cv2.imshow("🦾 YOLO Live Feed [1-4 to switch models, q to quit]", annotated)

    # 🔁 Ovládanie klávesmi
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in model_paths and key != current_key:
        current_key = key
        current_model_name = model_paths[current_key]
        print(f"🔁 Prepínam model na: {current_model_name}")
        model = YOLO(current_model_name)
        model.to("cpu")
        model.eval()

# 🧹 Cleanup
cap.release()
cv2.destroyAllWindows()
