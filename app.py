import streamlit as st
from vision_pipeline import analyze_image
from openai import OpenAI
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

client = OpenAI()

st.title("Vision + LLM Demo 🚀")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # save file tạm

    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    # chạy YOLO + vẽ box
    results = yolo_model("temp.jpg")

    # lấy ảnh đã vẽ box
    annotated_frame = results[0].plot()

    # hiển thị luôn
    st.image(annotated_frame, caption="Detected Image")
    # phân tích ảnh
    data = analyze_image("temp.jpg")

    st.subheader("Objects")
    st.write(data["count"])

    st.subheader("Caption")
    st.write(data["caption"])

    # hỏi đáp
    question = st.text_input("Ask a question about the image")

    if question:
        prompt = f"""
Image info:
Objects: {data['count']}
Caption: {data['caption']}

Question: {question}
Answer clearly.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content
        st.subheader("Answer")
        st.write(answer)