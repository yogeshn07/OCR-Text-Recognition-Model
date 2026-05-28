# 📇 Visiting Card OCR Extraction System

## 🔍 Overview
The Visiting Card OCR Extraction System is a privacy-first backend application that extracts structured contact information from visiting card images using Optical Character Recognition (OCR) and intelligent text parsing techniques.

The system accepts an image of a business card and returns meaningful, machine-readable data such as name, designation, company, phone number, email address, website, physical address, detected language, and an overall confidence score.

The project is implemented as a RESTful API using FastAPI and runs completely offline, making it suitable for environments where data privacy is critical 🔐.

---

## 🎯 Objectives
- 🧾 Automate the extraction of contact details from visiting cards  
- ✍️ Reduce manual data entry and human errors  
- 🎨 Support different card layouts, fonts, and image qualities
- 📊 Provide a confidence score indicating extraction reliability  
- 🧩 Design a modular and extensible OCR pipeline  
- 🔒 Ensure offline, privacy-preserving execution  

---

## 🛠️ Technologies Used

### 💻 Programming Language
- Python 3.8+

### ⚙️ Backend Framework
- FastAPI  
- Uvicorn  

### 🧠 OCR Engines
- Tesseract OCR (pytesseract)  
- PaddleOCR  

### 🖼️ Image Processing
- Pillow (PIL)  
- OpenCV  

### 🧩 Text Processing & Utilities
- Regular Expressions (re)  
- langdetect  
- phonenumbers  
- NumPy

---

## 🏗️ System Architecture
The system follows a pipeline-based architecture where each stage performs a specific task.

**Flow:**  
Image Input → Image Preprocessing → OCR Engine → Text Parsing → Confidence Calculation → JSON Output

**Detailed Steps:**
1. The client uploads a visiting card image (multipart or base64).
2. The image is preprocessed to improve OCR accuracy.
3. OCR is performed using Tesseract or PaddleOCR.
4. Raw text is parsed to extract structured fields.
5. A confidence score is calculated.
6. The final structured JSON response is returned.

---

## 📂 Project Folder Structure

Visiting_Card_OCR  
├── main.py – Core OCR pipeline using Tesseract OCR  
├── main_fixed.py – Improved and more robust parsing logic  
├── main_paddleocr.py – PaddleOCR-based advanced OCR pipeline  
├── requirements.txt – Python dependencies list  
├── README.md – Project documentation  
└── sample_images  
  ├── olivia.jpg – Light-themed visiting card test image  
  └── sandra.jpg – Dark-themed visiting card test image  

---

## ⚙️ Installation and Setup Guidelines

### 📌 Prerequisites
- Python 3.8 or higher  
- Tesseract OCR installed locally  
- Basic knowledge of FastAPI and Python  

### 🛠️ Setup Steps
1. Clone the project repository from GitHub.
2. Create and activate a Python virtual environment.
3. Install required dependencies using the requirements file.
4. Install Tesseract OCR and note its installation path.
5. Configure the Tesseract executable path inside the code if required.

---

## 🚀 Running the FastAPI Server
The API server can be started using:
- ▶️ Running the main Python file directly  
- ⚡ Using Uvicorn as the ASGI server  

Once running:
- Swagger UI: http://127.0.0.1:8000/docs  
- ReDoc: http://127.0.0.1:8000/redoc  

---

## 🔌 API Usage

### 📍 Endpoint
POST /extract

### 📥 Input Parameters
- Visiting card image (multipart file upload)  
- Base64-encoded image (optional)  
- OCR language parameter (default: eng)  

### 📤 API Response
The API returns a structured JSON object containing:
- 👤 Name  
- 🧑‍💼 Designation  
- 🏢 Company  
- 📞 Phone numbers  
- 📧 Email addresses  
- 🌍 Websites  
- 📍 Physical address  
- 🗣️ Detected language  
- 📊 Confidence score  
- 📝 Raw OCR text  
- ⚠️ Notes  

---

## 🧠 Developer Guidance & Improvements

### 📖 Understanding the Project
- Start with main.py to understand the OCR pipeline.
- Study the OCR execution logic to see how text is extracted.
- Follow the parsing logic to understand field extraction.
- Test the system using sample images before making changes.

### 🚀 Improvement Ideas
- Replace rule-based parsing with ML-based Named Entity Recognition (NER).
- Introduce layout-aware OCR models such as LayoutLM or DocTR.
- Improve multilingual OCR support.
- Add database integration for storing extracted contacts.
- Build a frontend interface for image upload and verification.
- Enable batch processing of visiting card images.

---

## 🔮 Future Scope
- 📱 Mobile camera-based visiting card scanning  
- ⏱️ Real-time OCR processing  
- 📇 CRM and contact management integration  
- ⚡ GPU-accelerated OCR pipelines  
- ☁️ Cloud deployment with scalability  
- 🏢 Enterprise-grade document digitization  

---

## 🌍 Real-World Applications
- 🏢 Corporate contact digitization  
- 📈 Sales and marketing lead management  
- 🤝 Conferences and professional networking events  
- 👥 HR onboarding workflows  
- 📄 Document digitization platforms  

Offline execution ensures privacy and data security 🔐.

---

## ✅ Conclusion
The Visiting Card OCR Extraction System demonstrates a practical real-world application of computer vision and text processing techniques. By combining OCR engines, intelligent preprocessing, and structured parsing, the system effectively automates visiting card digitization.

It's modular architecture and extensibility make it suitable for academic projects, internships, and real-world deployment.

---

## 👤 Author
Yogesh N - 
Sri Sairam Engineering College, Chennai -
Computer Science / AI-ML
