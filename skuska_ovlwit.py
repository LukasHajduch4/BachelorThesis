import torch
import time
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# Cesta k testovaciemu obrázku
image_path = "C:/Users/Lukas Hajduch/Downloads/BachelorThesis/data/images/IMG_20250316_210542_324.jpg"  # ← Zmeň podľa potreby
image = Image.open(image_path).convert("RGB")

query_texts = ["book", "bottle", "cup"]

for device in ["cpu", "cuda"]:
    if device == "cuda" and not torch.cuda.is_available():
        print("[⚠] CUDA not available, skipping CUDA test.")
        continue

    print(f"\n🔍 Testing OWL-ViT on {device.upper()}...")

    # Inicializácia modelu
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    # Príprava vstupu
    inputs = processor(text=[query_texts], images=image, return_tensors="pt").to(device)
    target_sizes = torch.tensor([[image.height, image.width]], dtype=torch.float32).to(device)

    # Warmup (kvôli JIT)
    with torch.no_grad():
        model(**inputs)

    # Meranie času inferencie
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.time()

    # Postprocessing
    results = processor.post_process_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.03
    )[0]

    print(f"🕒 Time on {device.upper()}: {end - start:.2f} seconds")
    print(f"✅ Detected {len(results['boxes'])} objects.")
