from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

img = "AI_vision/img.png"
image = Image.open(img)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

pixel_values = processor(images=image, return_tensors='pt').pixel_values
generated_ids = model.generate(pixel_values)

generatedText = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(generatedText)