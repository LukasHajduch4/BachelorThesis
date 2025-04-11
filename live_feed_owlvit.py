import cv2
import torch
import numpy as np
import time
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Inicializácia modelu
device = "cuda"
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# Definuj, čo hľadať (open-vocabulary)
query_texts = ["teddy bear"]

# Inicializuj kameru
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera sa nedá otvoriť")
    exit()

print("✅ OWL-ViT live feed spustený. Stlač Q pre ukončenie.")
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Snímok sa nepodarilo získať")
        break

    # FPS výpočet
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # OpenCV -> PIL Image
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Príprava vstupu pre OWL-ViT
    inputs = processor(text=[query_texts], images=image_pil, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Rozmery obrázku
    target_sizes = torch.tensor([(image_pil.height, image_pil.width)], dtype=torch.float32).to(device)

    # Výstupy z modelu
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.05  # môžeš zmeniť podľa potreby
    )[0]

    # Vykreslenie detekcií
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        if score < 0.05:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        label_text = f"{query_texts[label]}: {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # FPS overlay
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("OWL-ViT Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
