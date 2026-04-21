from openai import OpenAI
from vision_pipeline import analyze_image

client = OpenAI()

def ask_image(image_path, question):
    data = analyze_image(image_path)

    prompt = f"""
You are an AI assistant that answers questions about images.

Image information:
- Objects: {data['count']}
- Caption: {data['caption']}

Question: {question}

Answer clearly and concisely.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# test
if __name__ == "__main__":
    img = "test1.jpeg"

    while True:
        q = input("Ask a question: ")
        if q == "exit":
            break

        answer = ask_image(img, q)
        print("AI:", answer)