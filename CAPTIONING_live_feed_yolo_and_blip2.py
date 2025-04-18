import cv2
import torch
import time
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np

# 💻 Nastavenie zariadenia na CPU
device = torch.device("cpu")
print("✅ Používam zariadenie:", device)

# 🎯 Načítanie YOLOv8 modelu
yolo_model = YOLO('yolov8n.pt')
yolo_model.to(device)
yolo_model.eval()

# 🧠 Načítanie BLIP2 modelu a procesora
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_model.eval()

# 📷 Inicializácia kamery
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# 🔁 Slučka pre video
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("❌ Chyba pri načítaní rámca.")
        break

    # 🧠 YOLO detekcia
    results = yolo_model(frame)
    detections = results[0].boxes

    # 🖼️ Príprava obrázku pre PIL
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    for box in detections:
        # Získanie súradníc a orez
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image_pil.crop((x1, y1, x2, y2))

        # 📝 Popis objektu pomocou BLIP
        inputs = blip_processor(images=cropped, return_tensors="pt").to(device)

        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=20)
        description = blip_processor.decode(out[0], skip_special_tokens=True)

        # 🔲 Kreslenie boxu a popisu
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, description, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)

    # 🔢 FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 📺 Zobrazenie
    cv2.imshow("YOLOv8 + BLIP2 CPU", frame)

    # ⏹️ Ukončenie
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🧹 Upratanie
cap.release()
cv2.destroyAllWindows()
