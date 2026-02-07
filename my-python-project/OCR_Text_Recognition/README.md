# Visiting Card OCR Extractor (privacy-first)

This small FastAPI app accepts a visiting-card image (multipart upload or base64) and returns structured JSON with extracted fields (name, mobile, email, company, address, etc.).

Key points
- Runs on-device (no cloud) using Tesseract OCR (pytesseract). Install Tesseract binary separately.
- Phase 1: English supported. Language param accepts Tesseract language codes (e.g., `eng`, `tam`).
- Returns `raw_text` and a `confidence` score for audit.

Setup (Windows)

1. Install Python 3.8+ and create a virtual environment.

2. Install Tesseract OCR (binary):
   - Download from: https://github.com/tesseract-ocr/tesseract/releases
   - Install and add the Tesseract installation directory to your PATH (e.g., `C:\Program Files\Tesseract-OCR`).

3. Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Run the app:

```powershell
python main.py
# or
uvicorn main:app --reload --port 8000
```

API

POST /extract
- Body: multipart file `file` OR form field `image_base64` (base64 string). Optional form field `languages` (e.g. `eng` or `eng+tam`).

Response example (abbreviated):

```json
{
  "name": "Ravi Kumar",
  "designation": "Deputy Manager - Software Development",
  "company": "ABC Poultry Pvt Ltd",
  "mobile": ["+919876543210"],
  "email": ["ravi.kumar@abcpoultry.com"],
  "address": "123 Avinashi Road, Coimbatore, Tamil Nadu 641004",
  "language_detected": ["en"],
  "confidence": 0.86,
  "raw_text": "...",
  "notes": []
}
```

Notes & next steps
- To improve name/company extraction consider an NER model (spaCy) or a small fine-tuned classifier.
- Phase 2: add Tesseract language packs for Tamil/Hindi or use EasyOCR for Indic script improvements.
