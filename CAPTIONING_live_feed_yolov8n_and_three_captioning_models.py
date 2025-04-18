import cv2
import torch
import time
from ultralytics import YOLO
from PIL import Image
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer,
    AutoProcessor, GitForCausalLM
)

device = torch.device("cpu")

# YOLOv8n – rýchly detektor
yolo_model = YOLO("yolov8n.pt").to(device)
yolo_model.eval()

# Captioning modely
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

vitgpt2_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
vitgpt2_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vitgpt2_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

git_model = GitForCausalLM.from_pretrained("microsoft/git-base").to(device)
git_processor = AutoProcessor.from_pretrained("microsoft/git-base")

captioning_models = {
    "1": ("BLIP", blip_model, blip_processor),
    "2": ("ViT-GPT2", vitgpt2_model, (vitgpt2_processor, vitgpt2_tokenizer)),
    "3": ("GIT", git_model, git_processor),
}

current_key = "1"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Kamera sa nepodarilo spustiť.")
    exit()

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()

    for i, (box, conf) in enumerate(zip(detections, confidences)):
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        model_name, caption_model, processor = captioning_models[current_key]

        with torch.no_grad():
            if model_name == "BLIP":
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                out = caption_model.generate(**inputs, max_length=50)
                caption = processor.decode(out[0], skip_special_tokens=True)

            elif model_name == "ViT-GPT2":
                pixel_values = processor[0](images=pil_image, return_tensors="pt").pixel_values.to(device)
                out = caption_model.generate(pixel_values, max_length=50)
                caption = processor[1].decode(out[0], skip_special_tokens=True)

            elif model_name == "GIT":
                inputs = processor(images=pil_image, return_tensors="pt").to(device)
                out = caption_model.generate(**inputs, max_length=50)
                caption = processor.tokenizer.decode(out[0], skip_special_tokens=True)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, caption, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    fps = 1 / (time.time() - start_time)
    model_display = captioning_models[current_key][0]
    cv2.putText(frame, f"FPS: {fps:.2f} | Model: {model_display}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 + Captioning (1-3 to switch)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif chr(key) in captioning_models:
        current_key = chr(key)
        print(f"🔁 Prepínam na model: {captioning_models[current_key][0]}")

cap.release()
cv2.destroyAllWindows()
