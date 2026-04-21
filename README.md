# 🚀 Vision + LLM Image QA System

An end-to-end **multimodal AI application** that combines **Computer Vision** and **Large Language Models (LLMs)** to understand images and answer natural language questions.

---

## 🔥 Demo

Upload an image → AI will:

* Detect objects (YOLOv8)
* Generate image captions (BLIP)
* Answer questions about the image (LLM)

> 💡 Example:

```json
{
  "objects": {"person": 2, "dog": 1},
  "caption": "two people sitting with a dog",
  "answer": "There are 2 people in the image."
}
```

---

## 🧠 Key Features

* 🔍 **Object Detection** using YOLOv8
* 📝 **Image Captioning** using BLIP
* 🤖 **Question Answering** using LLM (OpenAI API)
* 🌐 **Interactive Web App** built with Streamlit
* ⚡ Real-time image analysis and reasoning

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Ultralytics YOLOv8
* HuggingFace Transformers (BLIP)
* OpenAI API (LLM)
* Streamlit

---

## 🔄 System Pipeline

```
Image 
  ↓
YOLOv8 (Object Detection)
  ↓
Extracted Objects + Count
  ↓
BLIP (Image Captioning)
  ↓
Caption + Structured Data
  ↓
LLM (Question Answering)
  ↓
Final Answer
```

---

## 🚀 How to Run Locally

### 1. Clone repo

```bash
git clone https://github.com/yourusername/vision-llm-image-qa.git
cd vision-llm-image-qa
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set API key

```bash
export OPENAI_API_KEY="your_api_key"
```

### 5. Run app

```bash
python -m streamlit run app.py
```

---

## 📸 Screenshots

👉 (Add your demo screenshot here)

---

## 🧠 Technical Highlights

* Applied **confidence thresholding** to filter low-quality detections from YOLO
* Combined **object detection + captioning** to improve semantic understanding
* Designed a **multimodal pipeline** to bridge vision and language reasoning
* Optimized inference by loading models only once

---

## ⚠️ Limitations

* May miss **small or occluded objects** due to dataset limitations
* Performance affected by **lighting conditions and image quality**
* LLM answers depend on extracted features (not raw image input)

---

## 🚀 Future Improvements

* Add OCR for text recognition
* Support video input + object tracking
* Fine-tune YOLO on custom dataset
* Improve UI/UX for better interaction

---

## 📌 Conclusion

This project demonstrates how to build a **real-world AI system** by integrating:

* Computer Vision (perception)
* Language Models (reasoning)

👉 A practical step toward building systems similar to **GPT-4 Vision**.

---

## 👨‍💻 Author

* Your Name
* GitHub: https://github.com/yourusername

---

⭐ If you find this project useful, feel free to star the repo!
