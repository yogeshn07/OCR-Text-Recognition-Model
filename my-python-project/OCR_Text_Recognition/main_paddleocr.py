from typing import List, Optional, Dict, Any
import base64
import io
import re

from fastapi import FastAPI, File, UploadFile, Form
from PIL import Image, ImageOps, ImageFilter
from paddleocr import PaddleOCR

# Initialize PaddleOCR lazily
ocr_engine = None

def get_ocr_engine():
    """Lazily initialize PaddleOCR engine on first use"""
    global ocr_engine
    if ocr_engine is None:
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
    return ocr_engine

app = FastAPI(title="Visiting Card OCR Extractor")


def _preprocess_image(img: Image.Image) -> Image.Image:
    """Preprocess image: convert to grayscale, autocontrast, sharpen"""
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
    except Exception:
        pass
    return img


def ocr_from_bytes(image_bytes: bytes, languages: List[str]) -> tuple[str, float, List[str]]:
    """Run PaddleOCR OCR. Returns (raw_text, ocr_confidence, notes)."""
    raw_text = ""
    notes: List[str] = []
    ocr_conf = 0.0
    try:
        import numpy as np
        import cv2
        
        # Read image using OpenCV for better compatibility
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv is None:
            # Fallback to PIL if OpenCV fails
            img = Image.open(io.BytesIO(image_bytes))
            img = _preprocess_image(img)
            img_cv = np.array(img)
        
        try:
            # Run PaddleOCR
            engine = get_ocr_engine()
            result = engine.ocr(img_cv)
            
            # Extract text and confidence scores
            text_parts = []
            confidences = []
            
            if result:
                for line_idx, line in enumerate(result):
                    if line:
                        for word_info in line:
                            # PaddleOCR returns: ((x1,y1), (x2,y2), ...), (text, confidence)
                            # or sometimes: [points, text, confidence]
                            try:
                                if isinstance(word_info, (list, tuple)):
                                    if len(word_info) >= 2:
                                        # Extract text and confidence
                                        if isinstance(word_info[1], tuple) and len(word_info[1]) == 2:
                                            # Format: (points, (text, confidence))
                                            text = word_info[1][0]
                                            conf = float(word_info[1][1])
                                        elif isinstance(word_info[1], str):
                                            # Format: (points, text, confidence)
                                            text = word_info[1]
                                            conf = float(word_info[2]) if len(word_info) > 2 else 0.5
                                        else:
                                            continue
                                        
                                        if text and text.strip():
                                            text_parts.append(text.strip())
                                            if 0 <= conf <= 1:
                                                confidences.append(conf)
                            except (ValueError, TypeError, IndexError):
                                continue
            
            raw_text = "\n".join(text_parts) if text_parts else ""
            
            # Calculate average confidence
            if confidences:
                ocr_conf = sum(confidences) / len(confidences)
            else:
                ocr_conf = 0.7 if raw_text else 0.0
                
        except Exception as e:
            notes.append(f"PaddleOCR failed: {str(e)}")
            raw_text = ""
            ocr_conf = 0.0
            
    except Exception as e:
        notes.append(f"Image processing or PaddleOCR failed: {str(e)}")
        raw_text = ""
        ocr_conf = 0.0

    return raw_text, ocr_conf, notes

def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text (lowercase, deduplicated)."""
    regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(regex, text)
    return [e.lower() for e in dict.fromkeys(emails)]


def extract_websites(text: str) -> List[str]:
    """Extract website URLs from text."""
    regex = r"((?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s,]*)?)"
    sites = re.findall(regex, text)
    normalized = []
    for s in dict.fromkeys(sites):
        s = s.strip().strip('.,')
        if not s.startswith("http"):
            s = "http://" + s
        normalized.append(s)
    return normalized


def extract_phones(text: str, default_region: str = "IN") -> List[str]:
    """Extract phone numbers and normalize to E.164 format."""
    phone_regex = r"(\+?\d[\d\s().-]{6,}\d)"
    candidates = re.findall(phone_regex, text)
    nums: List[str] = []
    try:
        import phonenumbers
        from phonenumbers import NumberParseException
        seen = set()
        for c in candidates:
            s = re.sub(r"[\s()\.-]", "", c)
            try:
                pn = phonenumbers.parse(s, default_region)
                if phonenumbers.is_valid_number(pn):
                    e = phonenumbers.format_number(pn, phonenumbers.PhoneNumberFormat.E164)
                    if e not in seen:
                        nums.append(e)
                        seen.add(e)
            except NumberParseException:
                continue
    except Exception:
        # Fallback: simple heuristic parsing
        seen = set()
        for c in candidates:
            s = re.sub(r"\D", "", c)
            if len(s) >= 7:
                if len(s) == 10:
                    e = "+91" + s
                elif s.startswith("91") and len(s) == 12:
                    e = "+" + s
                elif s.startswith("0") and len(s) == 11:
                    e = "+91" + s[1:]
                else:
                    e = "+" + s
                if e not in seen:
                    nums.append(e)
                    seen.add(e)
    return nums


def pick_name_and_company(
    lines: List[str], emails: List[str], phones: List[str], websites: List[str]
) -> Dict[str, Optional[str]]:
    """Extract name, company, and designation from text lines."""
    clean_lines = [l.strip() for l in lines if l and not any(k in l for k in emails + phones + websites)]
    name = None
    company = None
    designation = None

    # Try to find a name in first few lines (2-4 words, mostly alphabetic)
    for i, l in enumerate(clean_lines[:6]):
        words = l.split()
        if len(words) >= 2 and len(words) <= 4 and sum(1 for w in words if any(c.isalpha() for c in w)) >= 2:
            name = l
            if i + 1 < len(clean_lines):
                nxt = clean_lines[i + 1]
                if not re.search(r"\b(Pvt|Ltd|LLP|Inc|Company|Solutions|Technologies)\b", nxt, re.I):
                    designation = nxt
            break

    # Find company by looking for company keywords
    for l in clean_lines:
        if re.search(r"\b(Pvt|Private|Limited|Ltd|LLP|Inc|Corporation|Company)\b", l, re.I):
            company = l
            break

    # If no company found, use a line with mostly uppercase words
    if not company:
        for l in clean_lines:
            words = l.split()
            upcount = sum(1 for w in words if w.isupper() and len(w) > 1)
            if upcount >= 1 and len(words) >= 2:
                company = l
                break

    return {"name": name, "company": company, "designation": designation}


def extract_address(
    lines: List[str], emails: List[str], phones: List[str], websites: List[str]
) -> Optional[str]:
    """Extract address by looking for keywords and patterns."""
    address_parts = []
    address_keywords = ["road", "street", "rd", "lane", "city", "state", "district", "pin", "pincode", 
                       "india", "town", "village", "colony"]
    for l in lines:
        low = l.lower()
        # Match lines with address keywords, pin codes, or comma-separated fields
        if any(k in low for k in address_keywords) or re.search(r"\d{5,}", l) or ("," in l and len(l) > 20):
            address_parts.append(l.strip().strip('.,'))
    if address_parts:
        return ", ".join(address_parts)
    return None


def detect_languages(text: str) -> List[str]:
    """Detect languages in text using langdetect."""
    try:
        from langdetect import detect_langs
        langs = detect_langs(text)
        return [l.lang for l in langs[:3]]
    except Exception:
        return []


def parse_text(raw_text: str, languages: List[str]) -> Dict[str, Any]:
    """Parse extracted text and return normalized JSON with structured fields."""
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    emails = extract_emails(raw_text)
    websites = extract_websites(raw_text)
    phones = extract_phones(raw_text)
    name_company = pick_name_and_company(lines, emails, phones, websites)
    address = extract_address(lines, emails, phones, websites)
    langs = detect_languages(raw_text) if raw_text else []

    # Calculate confidence based on fields extracted
    core_found = sum(1 for v in [name_company.get("name"), phones, emails, address] if v)
    fields_score = core_found / 4.0

    result: Dict[str, Any] = {
        "name": name_company.get("name"),
        "designation": name_company.get("designation"),
        "company": name_company.get("company"),
        "mobile": phones,
        "email": emails,
        "address": address,
        "website": websites,
        "social": {},
        "extras": {},
        "language_detected": langs or languages,
        "confidence": round(0.5 * fields_score + 0.5 * 0.5, 2),
        "raw_text": raw_text,
        "notes": [],
    }
    return result


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    image_base64: Optional[str] = Form(None),
    languages: str = Form("eng"),
):
    """
    Extract text and structured data from a visiting card image using PaddleOCR.
    
    Accepts:
    - file: Multipart file upload (preferred)
    - image_base64: Optional base64-encoded image fallback
    - languages: OCR language(s), comma/plus separated (e.g. 'eng' or 'eng+hin')
    
    Returns normalized JSON with name, phone, email, address, website, confidence, etc.
    """
    lang_list = languages.split("+") if isinstance(languages, str) else (languages or ["eng"])
    image_bytes: Optional[bytes] = None
    notes: List[str] = []

    # Priority: file upload > base64 form
    if file is not None:
        image_bytes = await file.read()
    elif image_base64:
        try:
            # Remove data URI header if present
            header_removed = re.sub(r"^data:image/[^;]+;base64,", "", image_base64)
            image_bytes = base64.b64decode(header_removed)
        except Exception:
            return {"error": "invalid base64 image"}
    else:
        return {"error": "no image provided"}

    # Run OCR
    raw_text, ocr_conf, ocr_notes = ocr_from_bytes(image_bytes, lang_list)
    if ocr_notes:
        notes.extend(ocr_notes)

    # Parse text
    parsed = parse_text(raw_text, lang_list)
    parsed_conf = parsed.get("confidence", 0.5)

    # Combine OCR and parsing confidence
    final_conf = round(min(1.0, 0.7 * ocr_conf + 0.3 * parsed_conf), 2) if ocr_conf > 0 else parsed_conf
    parsed["confidence"] = final_conf
    parsed["raw_text"] = raw_text
    if notes:
        parsed.setdefault("notes", []).extend(notes)

    return parsed


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)