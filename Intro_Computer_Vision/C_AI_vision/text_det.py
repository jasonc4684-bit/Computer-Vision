from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

img = "AI_vision/img.png"
image = Image.open(img)

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

inputs = processor(text="<OCR_WITH_REGION>", images=image, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=1024)

generatedText = processor.batch_decode(outputs, skip_special_tokens=False)[0]

print(generatedText)