import cv2
import torch
import torchvision
import time
from torchvision.transforms import functional as F

# 💻 Zariadenie
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Používam zariadenie:", device)

# 📦 Načítaj predtrénovaný model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

# 🏷️ COCO labely
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 📷 Kamera
cap = cv2.VideoCapture(0)

# 🔄 Hlavný cyklus
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("❌ Zlyhalo načítanie snímky.")
        break

    frame = cv2.resize(frame, (640, 480))
    image_tensor = F.to_tensor(frame).to(device)

    with torch.no_grad():
        preds = model([image_tensor])[0]

    # 🧾 FPS výpočet
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # 🖍️ Vykreslenie detekcií
    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        if score > 0.5:
            x1, y1, x2, y2 = box.int().tolist()
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
            conf = round(score.item() * 100, 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf}%", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 🕒 Zobrazenie FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Faster R-CNN - Live", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 🧹 Upratovanie
cap.release()
cv2.destroyAllWindows()
