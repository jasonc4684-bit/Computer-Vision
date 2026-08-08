from PIL import Image
from transformers import Florence2ForConditionalGeneration, AutoProcessor

img = "C:/Users/yingl/OneDrive/Computer-Vision-Project-main/Intro_Computer_Vision/C_AI_vision/img.png"
image = Image.open(img).convert("RGB")

processor = AutoProcessor.from_pretrained("Florence-community/Florence-2-base")
model = Florence2ForConditionalGeneration.from_pretrained("Florence-community/Florence-2-base")

inputs = processor(text="<OCR_WITH_REGION>", images=image, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=1024, num_beams=3)

generatedText = processor.batch_decode(outputs, skip_special_tokens=False)[0]

result = processor.post_process_generation(generatedText, task="<OCR_WITH_REGION>", 
                                            image_size=(image.width, image.height))

for label, box in zip(result["<OCR_WITH_REGION>"]["labels"], 
                        result["<OCR_WITH_REGION>"]["quad_boxes"]):
    print(label.replace("</s>", "").strip(), box)