from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model 1 lần (rất quan trọng)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)


def generate_caption(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)

        out = model.generate(**inputs, max_new_tokens=50)

        caption = processor.decode(out[0], skip_special_tokens=True)

        return caption

    except Exception as e:
        return f"Error generating caption: {str(e)}"


# test
if __name__ == "__main__":
    print("Using device:", device)
    caption = generate_caption("data/samples/test1.jpeg")
    print("Caption:", caption)