from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

img = "AI_vision/img.png"
image = Image.open(img).convert("RGB")

pixel_values = processor(images=image, return_tensors='pt').pixel_values
generated_ids = model.generate(pixel_values)

generatedText = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generatedText)