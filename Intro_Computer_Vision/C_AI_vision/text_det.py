import cv2
from transformers import pipeline

pipe = pipeline("image-text-to-text", model="microsoft/trocr-base-printed")

img = "AI_vision/text.png"
result = pipe(img)

print(result[0]["generated_text"])