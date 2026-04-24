from core.caption import generate_caption
from core.vqa import answer_question


def run_pipeline(image_path: str, question: str):
    # Step 1: caption
    caption = generate_caption(image_path)

    # Step 2: QA
    answer = answer_question(caption, question)

    return {
        "caption": caption,
        "answer": answer
    }


# test
if __name__ == "__main__":
    image = "data/samples/test1.jpeg"
    question = "What is the animal doing?"

    result = run_pipeline(image, question)

    print("Caption:", result["caption"])
    print("Answer:", result["answer"])