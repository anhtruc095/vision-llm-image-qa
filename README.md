# Vision + LLM Image QA System 🚀

## 🔥 Demo
Upload an image and ask questions about it using AI.

## 🧠 Features
- Object Detection (YOLOv8)
- Image Captioning (BLIP)
- Question Answering (LLM)
- Web UI with Streamlit

## ⚙️ Tech Stack
- Python
- PyTorch
- Ultralytics YOLOv8
- HuggingFace Transformers
- OpenAI API
- Streamlit

## 🔄 Pipeline
Image → YOLO → Objects → BLIP → Caption → LLM → Answer

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
