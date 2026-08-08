from transformers import pipeline

pipe = pipeline("image-to-text", model="microsoft/trocr-base-printed")

img = "AI_vision\text.png"
result = pipe(img)

print(result[0]["generated_text"])