# 🧠 Vision Chat Assistant (YOLO + BLIP + LLM)

An end-to-end **multimodal AI system** that combines **Computer Vision** and **Large Language Models (LLMs)** to understand images and answer natural language questions — with an interactive **ChatGPT-style UI**.

---

## 🚀 Demo

Upload an image → Ask a question → Get:

* 🎯 **Highlighted objects** based on your question
* 💬 **Natural language answers** (streaming like ChatGPT)
* 🧠 **Context-aware reasoning**

---

## ✨ Key Features

* 🔍 **Object Detection** using YOLOv8
* 📝 **Image Captioning** using BLIP
* 🤖 **Question Answering** using LLM (OpenAI API)
* 💬 **Chat UI (multi-turn)** like ChatGPT
* ✍️ **Streaming responses** (typing effect)
* 🎯 **Query-aware object highlighting**
* 🎨 Clean and modern UI with Streamlit

---

## 🧠 System Architecture

```
Image
 ├── YOLOv8 → Object Detection
 ├── BLIP → Caption Generation
 └── LLM → Question Answering
```

---

## 📸 Example

**User Question:**

```
Where is the dog?
```

**System Output:**

* Highlights the dog in the image 🟩
* Returns a clear, natural language answer

---

## ⚙️ Tech Stack

* Python
* PyTorch
* Ultralytics YOLOv8
* HuggingFace Transformers (BLIP)
* OpenAI API (LLM)
* Streamlit

---

## 🔄 Full Pipeline

```
Image 
  ↓
YOLOv8 (Object Detection)
  ↓
Filtered Objects
  ↓
BLIP (Image Captioning)
  ↓
Caption + Context
  ↓
LLM (Reasoning & QA)
  ↓
Final Answer + Highlight
```

---

## 🚀 How to Run Locally

### 1. Clone repository

```bash
git clone https://github.com/anhtruc095/vision-llm-image-qa.git
cd vision-llm-image-qa
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup API key

```bash
export OPENAI_API_KEY="your_api_key"
```

### 5. Run the app

```bash
streamlit run app.py
```

---

## 🧠 Technical Highlights

* Applied **confidence thresholding** to filter low-quality detections
* Designed **query-aware highlighting** (only show relevant objects)
* Combined **vision + language models** for better reasoning
* Built **ChatGPT-style UI with streaming responses**
* Optimized performance by **loading models once**

---

## ⚠️ Limitations

* May miss **small or occluded objects**
* Performance depends on **image quality & lighting**
* LLM answers rely on extracted features (not raw vision)

---

## 🚀 Future Improvements

* 🔥 Semantic matching (e.g., “animal” → dog)
* 💬 Long-term conversational memory
* 🎯 Attention heatmap visualization
* ⚡ Faster inference (quantization / optimization)
* 🌐 Deploy to cloud (Streamlit / Docker)

---

## 💼 Use Cases

* AI-powered visual assistants
* Image search & understanding
* Smart surveillance systems
* Human-AI interaction research

---

## 📁 Project Structure

```
vision-llm-image-qa/
│
├── core/
│   ├── caption.py
│   ├── vqa.py
│   └── pipeline.py
│
├── data/
│   └── samples/
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🖼️ Demo Preview

![demo](output.gif)


---

## 💼 Resume Highlights

* Built a multimodal AI system combining YOLOv8, BLIP, and LLMs
* Implemented query-aware object highlighting for explainability
* Developed a ChatGPT-style UI with real-time streaming responses
* Designed modular architecture for scalable AI applications

---

## 📌 Conclusion

This project demonstrates how to integrate:

* Computer Vision (perception)
* Language Models (reasoning)

👉 A practical step toward systems like **GPT-4 Vision**.

---

## 👨‍💻 Author

**Ton Anh Truc**
GitHub: https://github.com/anhtruc095

---

## ⭐ If you find this project useful

Give it a star ⭐ on GitHub!
