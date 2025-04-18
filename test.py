import cv2

if hasattr(cv2, 'freetype'):
    print("✅ cv2.freetype is available!")
else:
    print("❌ cv2.freetype is NOT available!")