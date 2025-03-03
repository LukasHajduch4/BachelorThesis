import requests
from PIL import Image, ImageDraw
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load OWL-ViT model and processor
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# Load image from local path
image_path = r"C:\\Users\\Lukas Hajduch\\OneDrive\\Dokumenty\\Bachelor thesis\\BachelorThesis\\data\\images\\20250225_191040.jpg"
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Define query texts
text_queries = ["bottle", "cup", "book"]

# Process image and text inputs
inputs = processor(text=[text_queries], images=image, return_tensors="pt").to(device)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Target image sizes (height, width) to rescale box predictions
target_sizes = torch.tensor([(image.height, image.width)], dtype=torch.float32).to(device)

# Convert outputs to bounding boxes and class logits
results = processor.post_process_object_detection(
    outputs=outputs, target_sizes=target_sizes, threshold=0.05
)

# Retrieve predictions
result = results[0]
boxes, scores, labels = result["boxes"], result["scores"], result["labels"]

# Draw bounding boxes and labels
for box, score, label in zip(boxes, scores, labels):
    box = [round(i, 2) for i in box.tolist()]
    print(f"Detected {text_queries[label]} with confidence {round(score.item(), 3)} at location {box}")
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1] - 10), f"{text_queries[label]}: {round(score.item(), 3)}", fill="red")

# Save or show the output image
output_path = "outputs/owlvit-detection-output.jpg"
image.save(output_path)
image.show()