from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-small"
).to(device)


def answer_question(caption: str, question: str) -> str:
    prompt = f"""
You are a vision-language assistant.

Image description: {caption}

Question: {question}

Answer briefly:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=50
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


# test
if __name__ == "__main__":
    caption = "a cat sitting on a sofa"
    question = "What animal is this?"

    print("Answer:", answer_question(caption, question))