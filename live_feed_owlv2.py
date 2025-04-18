import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# 1️⃣ Zariadenie
device = torch.device("cpu")
print("💻 Using device:", device)

# 2️⃣ Model a processor
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16").to(device)
model.eval()
if device.type == "cuda":
    model = model.half()

processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")

# 3️⃣ Textové prompt-y
texts = [["a cup", "a teddy bear"]]
dummy_image = Image.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
inputs = processor(text=texts, images=dummy_image, return_tensors="pt")

# ⚠️ Zachováme input_ids ako LongTensor, ostatné môžu byť float16
text_inputs = {
    k: (v.to(device).half() if k != "input_ids" else v.to(device))
    for k, v in inputs.items() if k != "pixel_values"
}

# 4️⃣ Kamera a nastavenie buffera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("📷 Kamera inicializovaná, spúšťame loop...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Frame capture failed.")
        break

    # 5️⃣ Zmenšenie rozlíšenia a konverzia
    resized_frame = cv2.resize(frame, (640, 480))
    pil_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))

    # 6️⃣ Len pixel_values pre daný frame
    image_inputs = processor(images=pil_image, return_tensors="pt")["pixel_values"]
    image_inputs = image_inputs.to(device).half() if device.type == "cuda" else image_inputs.to(device)

    # 7️⃣ Spojenie s textovými embeddingmi
    inputs = {**text_inputs, "pixel_values": image_inputs}

    # 8️⃣ Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # 9️⃣ Výstup
    target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)
    
    boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

    for box, score, label in zip(boxes, scores, labels):
        if score < 0.2:
            continue

        box = box.to("cpu").numpy().astype(int)
        label_text = texts[0][label]
        conf = round(score.item() * 100, 1)

        cv2.rectangle(resized_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(resized_frame, f"{label_text} {conf}%", (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 🔁 Výstupný frame
    cv2.imshow("🦉 OWLv2 Live Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 Ukončujem...")
        break

cap.release()
cv2.destroyAllWindows()
