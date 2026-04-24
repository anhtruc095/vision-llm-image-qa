import streamlit as st
from core.pipeline import run_pipeline
from ultralytics import YOLO
from openai import OpenAI
import cv2
import time

# -------- INIT --------
client = OpenAI()
yolo_model = YOLO("yolov8n.pt")

st.set_page_config(page_title="Vision Chat", layout="centered")
st.title("🧠 Vision Chat Assistant")

# -------- SESSION --------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "image_path" not in st.session_state:
    st.session_state.image_path = None

# -------- FUNCTIONS --------
def match_objects(question, names):
    question = question.lower()
    return [name for name in names if name in question]

def need_caption(question):
    question = question.lower()
    keywords = ["describe", "what is in", "caption", "tell me about"]
    return any(k in question for k in keywords)

def stream_text(text):
    for i in range(len(text)):
        yield text[:i+1]
        time.sleep(0.01)  # chỉnh tốc độ gõ ở đây

# -------- UPLOAD --------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    temp_path = "data/samples/temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.image_path = temp_path

# -------- SHOW IMAGE --------
if st.session_state.image_path:
    st.image(st.session_state.image_path, width="stretch")

# -------- SHOW CHAT HISTORY --------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------- INPUT --------
if prompt := st.chat_input("Ask something about the image..."):

    # user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    temp_path = st.session_state.image_path

    # YOLO detect
    results = yolo_model(temp_path, conf=0.5)
    boxes = results[0].boxes
    names = yolo_model.names

    matched_objects = match_objects(prompt, names.values())
    image_cv = cv2.imread(temp_path)

    # -------- HIGHLIGHT --------
    if matched_objects:
        for box in boxes:
            cls_id = int(box.cls[0])
            name = names[cls_id]

            if name in matched_objects:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(
                    image_cv,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

        st.image(image_cv, width="stretch")
    else:
        st.image(results[0].plot(), width="stretch")

    # -------- AI RESPONSE --------
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            result = run_pipeline(temp_path, prompt)

            prompt_llm = f"""
Image description: {result['caption']}
User question: {prompt}

Answer naturally and clearly.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt_llm}]
            )

            answer = response.choices[0].message.content

        # -------- STREAMING EFFECT --------
        placeholder = st.empty()
        full_text = ""

        for chunk in stream_text(answer):
            full_text = chunk
            placeholder.markdown(full_text + "▌")

        placeholder.markdown(full_text)

        # caption nếu cần
        if need_caption(prompt):
            st.caption(result["caption"])

    # save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )