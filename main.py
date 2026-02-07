from typing import List, Optional, Dict, Any, Tuple
import base64
import io
import re
import urllib.parse

# PIL may not be installed in the environment used for quick local tests; guard it
try:
    from PIL import Image, ImageOps, ImageFilter
except Exception:
    Image = ImageOps = ImageFilter = None

# try to import FastAPI and pydantic, but allow importing this module even when
# they are not installed so parse_text can be used independently for tests
try:
    from fastapi import FastAPI, File, UploadFile, Form
except Exception:
    FastAPI = None
    File = UploadFile = Form = None

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = object
import io
import re
from PIL import Image, ImageOps, ImageFilter
import urllib.parse

app = FastAPI(title="Visiting Card OCR Extractor") if FastAPI is not None else None


class ExtractRequest(BaseModel):
    image_base64: Optional[str] = None
    languages: Optional[List[str]] = ["eng"]


def _preprocess_image(img) -> Any:
    # basic preprocessing: convert to grayscale, autocontrast and slight sharpen
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
    except Exception:
        pass
    return img


def ocr_from_bytes(image_bytes: bytes, languages: List[str]):
    # Try to use pytesseract if available; otherwise return empty string and note
    raw_text = ""
    notes = []
    try:
        import pytesseract
        img = Image.open(io.BytesIO(image_bytes))
        img = _preprocess_image(img)
        lang = "+".join(languages) if languages else "eng"
        try:
            raw_text = pytesseract.image_to_string(img, lang=lang)
            # try to compute a rough OCR confidence
            try:
                data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
                confs = [int(c) for c in data.get("conf", []) if c and int(c) >= 0]
                if confs:
                    ocr_conf = sum(confs) / (len(confs) * 100.0)
                else:
                    ocr_conf = 0.6
            except Exception:
                ocr_conf = 0.6
        except Exception:
            raw_text = pytesseract.image_to_string(img)
            ocr_conf = 0.6
    except Exception:
        notes.append("pytesseract not available or failed - OCR skipped")
        raw_text = ""
        ocr_conf = 0.0
    return raw_text, ocr_conf, notes


def extract_emails(text: str) -> List[str]:
    regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(regex, text)
    return [e.lower() for e in dict.fromkeys(emails)]


def extract_websites(text: str) -> List[str]:
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
    # find candidate phone strings
    phone_regex = r"(\+?\d[\d\s().-]{6,}\d)"
    candidates = re.findall(phone_regex, text)
    nums = []
    # try to use phonenumbers if available
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
        # fallback: keep digits; if 10 digits -> +91
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


def extract_gstin(text: str) -> Optional[str]:
    # GSTIN pattern: 15 chars
    m = re.search(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b", text)
    return m.group(0) if m else None


def extract_cin(text: str) -> Optional[str]:
    m = re.search(r"\bU[0-9A-Z]{5}[0-9]{4}[A-Z]{3}[0-9]{6}\b", text)
    return m.group(0) if m else None


def pick_name_and_company(lines: List[str], emails, phones, websites) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Heuristics: choose first non-contact line as name (>=2 words)
    clean_lines = [l.strip() for l in lines if l and not any(k in l for k in emails + phones + websites)]
    name = None
    company = None
    designation = None
    # pick name: first line with 2+ words and short (<5 words)
    for i, l in enumerate(clean_lines[:6]):
        words = l.split()
        if len(words) >= 2 and len(words) <= 4 and sum(1 for w in words if any(c.isalpha() for c in w)) >= 2:
            name = l
            # designation often next line
            if i + 1 < len(clean_lines):
                next_line = clean_lines[i + 1]
                # if next line not company-like, treat as designation
                if not re.search(r"\b(Pvt|Ltd|LLP|Inc|Company|Solutions|Technologies|Technologies)\b", next_line, re.I):
                    designation = next_line
            break

    # company: look for keywords or all-caps tokens
    for l in clean_lines:
        if re.search(r"\b(Pvt|Pvt. Ltd|Private|Limited|Ltd|LLP|Inc|Corporation|Company)\b", l, re.I):
            company = l
            break
    if not company:
        # fallback: a line that is mostly uppercase and has >1 word
        for l in clean_lines:
            words = l.split()
            upcount = sum(1 for w in words if w.isupper() and len(w) > 1)
            if upcount >= 1 and len(words) >= 2:
                company = l
                break

    return name, company, designation


def extract_address(lines: List[str], emails, phones, websites) -> Optional[str]:
    # Join consecutive lines that look like address (contain digits or address keywords)
    address_parts = []
    address_keywords = ["road", "street", "rd", "lane", "city", "state", "district", "pin", "pincode", "india", "town", "village", "colony"]
    for l in lines:
        low = l.lower()
        if any(k in low for k in address_keywords) or re.search(r"\d{5,}", l) or ("," in l and len(l) > 20):
            address_parts.append(l.strip().strip('.,'))
    if address_parts:
        return ", ".join(address_parts)
    return None


def detect_languages(text: str) -> List[str]:
    try:
        from langdetect import detect_langs
        langs = detect_langs(text)
        return [l.lang for l in langs[:3]]
    except Exception:
        return []


def parse_text(raw_text: str, languages: List[str] = ["eng"]) -> Dict[str, Any]:
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    emails = extract_emails(raw_text)
    websites = extract_websites(raw_text)
    phones = extract_phones(raw_text)
    gstin = extract_gstin(raw_text)
    cin = extract_cin(raw_text)
    name, company, designation = pick_name_and_company(lines, emails, phones, websites)
    address = extract_address(lines, emails, phones, websites)
    langs = detect_languages(raw_text) if raw_text else []

    # confidence: based on presence of core fields and a rough baseline
    core_found = sum(1 for v in [name, phones, emails, address] if v)
    fields_score = core_found / 4.0

    result = {
        "name": name,
        "designation": designation,
        "company": company,
        "mobile": phones,
        "email": emails,
        "address": address,
        "website": websites,
        "social": {},
        "extras": {},
        "language_detected": langs or languages,
        "confidence": round(0.5 * fields_score + 0.5 * 0.5, 2),  # placeholder until OCR conf is known
        "raw_text": raw_text,
        "notes": []
    }
    if gstin:
        result["extras"]["gstin"] = gstin
    if cin:
        result["extras"]["cin"] = cin
    return result


@app.post("/extract")
async def extract(file: Optional[Any] = File(None), image_base64: Optional[str] = Form(None), languages: Optional[str] = Form("eng")):
    """Accepts multipart file upload or base64 in form; languages is a + separated string like 'eng' or 'eng+tam'"""
    notes = []
    lang_list = languages.split("+") if isinstance(languages, str) else (languages or ["eng"])
    image_bytes = None
    if file is not None:
        image_bytes = await file.read()
    elif image_base64:
        try:
            header_removed = re.sub(r"^data:image/[^;]+;base64,", "", image_base64)
            image_bytes = base64.b64decode(header_removed)
        except Exception:
            return {"error": "invalid base64 image"}
    else:
        return {"error": "no image provided"}

    raw_text, ocr_conf, ocr_notes = ocr_from_bytes(image_bytes, lang_list)
    notes.extend(ocr_notes)
    parsed = parse_text(raw_text, lang_list)
    # combine confidences: use ocr_conf (0-1) and parsed confidence to produce final
    parsed_conf = parsed.get("confidence", 0.5)
    final_conf = round(min(1.0, 0.7 * ocr_conf + 0.3 * parsed_conf), 2) if ocr_conf > 0 else parsed_conf
    parsed["confidence"] = final_conf
    parsed["raw_text"] = raw_text
    if notes:
        parsed.setdefault("notes", []).extend(notes)
    return parsed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
