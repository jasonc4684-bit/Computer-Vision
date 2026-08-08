from transformers import pipeline

pipe = pipeline("image-to-text", model="microsoft/trocr-base-printed")


result = pipe()