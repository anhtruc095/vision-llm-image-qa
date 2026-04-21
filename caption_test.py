from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# load model (lần đầu sẽ download)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# load ảnh
image = Image.open("test1.jpeg")

# xử lý
inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)

# decode
caption = processor.decode(out[0], skip_special_tokens=True)

print("Caption:", caption)