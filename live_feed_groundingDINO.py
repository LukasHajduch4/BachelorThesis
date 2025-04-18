import cv2
import torch
import time
from torchvision import transforms
from groundingdino.util.inference import load_model, predict, annotate
from PIL import Image
import numpy as np

def preprocess_opencv_image(frame):
    # BGR (OpenCV) -> RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)

    # Transform ako v GroundingDINO
    transform = transforms.Compose([
        transforms.Resize((512, 512), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image_pil).unsqueeze(0)  # [1, 3, H, W]
    return image_pil, image_tensor

# Inicializácia zariadenia
device = torch.device("cpu")

# Načítanie modelu
model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "GroundingDINO/weights/groundingdino_swint_ogc.pth"
).to(device)
model.eval()
# Prompt
TEXT_PROMPT = "cup"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS výpočet
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # Predspracovanie
    image_source, image_tensor = preprocess_opencv_image(frame)
    image_tensor = image_tensor.to(device)

    # Predikcia
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # Annotácia výsledkov
    if hasattr(image_source, "mode"):  # it's a PIL.Image
        image_source = np.array(image_source)

    annotated_frame = annotate(
        image_source=image_source,
        boxes=boxes,
        logits=logits,
        phrases=phrases
    )

    # FPS overlay
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("GroundingDINO Live", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
