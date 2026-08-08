from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")


img = "AI_vision/img.png"
result = pipe(img)

print(result[0]["generated_text"])