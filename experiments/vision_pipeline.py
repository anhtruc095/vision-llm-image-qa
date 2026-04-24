from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from collections import Counter

# load model (CHỈ 1 LẦN)
yolo_model = YOLO("yolov8n.pt")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True
)

def analyze_image(image_path):
    # -------- YOLO --------
    results = yolo_model(image_path)
    objects = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            objects.append(yolo_model.names[cls])

    count = dict(Counter(objects))

    # -------- BLIP --------
    image = Image.open(image_path)

    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)

    caption = processor.decode(out[0], skip_special_tokens=True)

    return {
        "objects": objects,
        "count": count,
        "caption": caption
    }

# test
if __name__ == "__main__":
    result = analyze_image("test1.jpeg")  # nhớ đúng đuôi file
    #print(result)
    print("\nObjects:", result["objects"])
    print("Count:", result["count"])
    print("Caption:", result["caption"])