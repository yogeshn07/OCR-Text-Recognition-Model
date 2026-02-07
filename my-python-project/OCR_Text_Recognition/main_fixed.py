from typing import List, Optional, Dict, Any, Tuple
import base64
import io
import re

from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
from PIL import Image, ImageOps, ImageFilter

app = FastAPI(title="Visiting Card OCR Extractor")


def _preprocess_image(img) -> Any:
    """Preprocessing for better OCR results."""
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img)
        img = img.filter(ImageFilter.SHARPEN)
    except Exception:
        pass
    return img


def ocr_from_bytes(image_bytes: bytes, languages: List[str]):
    raw_text = ""
    notes = []
    try:
        import pytesseract
        # Configure pytesseract to use Tesseract binary
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        
        img = Image.open(io.BytesIO(image_bytes))
        img = _preprocess_image(img)
        lang = "+".join(languages) if languages else "eng"
        raw_text = pytesseract.image_to_string(img, lang=lang)
        ocr_conf = 0.6
        try:
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if c and int(c) >= 0]
            if confs:
                ocr_conf = sum(confs) / (len(confs) * 100.0)
        except Exception:
            pass
    except Exception as e:
        notes.append(f"pytesseract failed: {str(e)} - Trying PaddleOCR...")
        # Fallback to PaddleOCR
        try:
            import numpy as np
            import cv2
            from paddleocr import PaddleOCR
            
            nparr = np.frombuffer(image_bytes, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img_cv is not None:
                ocr = PaddleOCR(use_angle_cls=True, lang='en')
                result = ocr.ocr(img_cv)
                
                text_parts = []
                confidences = []
                
                if result:
                    for line in result:
                        if line:
                            for word_info in line:
                                try:
                                    if isinstance(word_info, (list, tuple)) and len(word_info) >= 2:
                                        if isinstance(word_info[1], tuple) and len(word_info[1]) == 2:
                                            text = word_info[1][0]
                                            conf = float(word_info[1][1])
                                        elif isinstance(word_info[1], str):
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
                
                raw_text = "\n".join(text_parts)
                ocr_conf = sum(confidences) / len(confidences) if confidences else 0.7 if raw_text else 0.0
                notes.append("Used PaddleOCR as fallback")
        except Exception as e2:
            notes.append(f"PaddleOCR also failed: {str(e2)}")
            raw_text = ""
            ocr_conf = 0.0
    
    return raw_text, ocr_conf, notes


def extract_emails(text: str) -> List[str]:
    # Primary regex for standard email format
    regex = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    emails = re.findall(regex, text)
    
    # Secondary pattern: look for email-like patterns with spaces or special chars that OCR might introduce
    # e.g., "sandra . tucker @ fine fx . com" or "sandra.tucker @ finefx.com"
    secondary_regex = r"[a-zA-Z0-9._+-]+\s*@\s*[a-zA-Z0-9\s.-]+\.[a-zA-Z]{2,}"
    secondary_matches = re.findall(secondary_regex, text, re.IGNORECASE)
    
    # Clean up secondary matches (remove extra spaces and fix common OCR errors)
    for match in secondary_matches:
        cleaned = re.sub(r'\s+', '', match).lower()
        if cleaned not in [e.lower() for e in emails]:
            emails.append(cleaned)
    
    # Remove duplicates and invalid entries
    final_emails = []
    seen = set()
    for e in emails:
        e_lower = e.lower()
        # Skip if it looks like phone@website (contains only digits before @)
        if e_lower not in seen and '@' in e:
            local_part = e.split('@')[0]
            # Must have at least some letters, not just digits
            if any(c.isalpha() for c in local_part):
                seen.add(e_lower)
                final_emails.append(e_lower)
    
    return final_emails


def extract_websites(text: str) -> List[str]:
    """Extract website URLs from text.

    Avoid matching tokens that are part of an email address, and deduplicate by base domain.
    Use finditer to examine surrounding characters so email-local-parts are ignored.
    """
    normalized = []
    seen_domains = set()
    
    # Primary regex: require at least one dot in domain part and allow multiple domain parts
    regex = r"((?:https?://)?(?:www\.)?[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+(?:/[^\s,]*)?)"
    
    for m in re.finditer(regex, text):
        s = m.group(1).strip().strip('.,')
        start, end = m.start(1), m.end(1)
        
        # Skip if adjacent to @ (part of email)
        if (start > 0 and text[start - 1] == '@') or (end < len(text) and text[end:end+1] == '@'):
            continue
        
        # Skip if this is a phone number pattern (lots of digits)
        if re.match(r'^[\d\s().-]{10,}$', s):
            continue

        if not s.startswith("http"):
            s = "http://" + s

        # Extract base domain (remove www.) for deduplication
        domain_match = re.search(r"https?://(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+)", s, re.I)
        if domain_match:
            base_domain = domain_match.group(1).lower()
            # Skip if domain ends with something that looks like an email local part (hello, test, etc.)
            # Check if it contains a single letter followed by domain (like .hello at end)
            if not re.search(r'\.[a-z]{2,}$', base_domain) or base_domain.count('.') == 1:
                if base_domain not in seen_domains:
                    normalized.append(s)
                    seen_domains.add(base_domain)

    return normalized


def extract_phones(text: str) -> List[str]:
    # Process each line separately to avoid matching across newlines
    nums = []
    seen = set()
    
    # Split text into lines and search for phones in each line separately
    for line in text.split('\n'):
        # Phone regex - match standard formats: 123-456-7890 or +1 123-456-7890
        phone_regex = r"(\+\d[\d\s().-]*\d|\d{3}[-.]\d{3}[-.]\d{4})"
        candidates = re.findall(phone_regex, line)
        
        for c in candidates:
            # Remove all non-digit characters
            digits_only = re.sub(r"\D", "", c)
            
            # Only accept if it's a reasonable phone length (7-15 digits)
            if 7 <= len(digits_only) <= 15:
                # Format with +
                formatted = "+" + digits_only
                
                if formatted not in seen:
                    nums.append(formatted)
                    seen.add(formatted)
    
    return nums


def parse_text(raw_text: str) -> Dict[str, Any]:
    # Split lines and detect raw contacts first
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    emails = extract_emails(raw_text)
    phones = extract_phones(raw_text)
    websites = extract_websites(raw_text)

    # Precompile helpful regexes for cleaning
    email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
    website_re = re.compile(r"https?://\S+|\bwww\.\S+\b", re.I)
    phone_re = re.compile(r"\+?\d[\d\s().-]{5,}\d")

    # Build cleaned lines where emails/phones/websites are removed but other context preserved
    cleaned_lines = []
    for l in lines:
        tmp = email_re.sub('', l)
        tmp = website_re.sub('', tmp)
        tmp = phone_re.sub('', tmp)
        tmp = re.sub(r'[-–—]|[_]{2,}', ' ', tmp)
        tmp = re.sub(r'\s{2,}', ' ', tmp).strip().strip(' ,.-@')
        if tmp:
            cleaned_lines.append(tmp)

    # Fallback to original lines if nothing remains
    if not cleaned_lines:
        cleaned_lines = lines

    name = None
    company = None
    designation = None
    address = None

    # Title keywords for designation detection
    title_keywords = r"\b(Real Estate Agent|manager|director|engineer|developer|consultant|head|lead|vp|vice|officer|sales|marketing|founder|co-founder|chief|president|analyst|associate|sr|jr|CEO|CTO|CFO|COO|MD|GM|SVP|VP|AVP|artist|designer|coordinator|specialist|admin|executive|art\s+director|agent|estate|real\s+estate)\b"
    
    # US State abbreviations to exclude (MD, CA, NY, etc.)
    state_abbreviations = r"\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b"
    
    # Skip lines that look like logos, special symbols, or gibberish
    skip_patterns = r"^[<>@#$%^&*()_+=\[\]{};:'\",.<>?/\\|-]*$|^[<>].*[<>]$"
    
    # Address detection patterns
    address_keywords = ["plot", "plot no", "road", "rdy", "road no", "street", "rd", "lane", "city", "state", "district", "pin", "pincode",
                        "india", "town", "village", "colony", "avenue", "boulevard", "drive", "ave", "midc", "andheri", "marol", "mumbai"]
    
    # First pass: look for name in cleaned lines BEFORE extracting from email
    for i, l in enumerate(cleaned_lines[:10]):
        # Skip lines with mostly special characters or starting with special chars
        if re.match(skip_patterns, l) or l.startswith('<') or l.startswith('>'):
            continue
            
        words = l.split()
        has_alpha = any(c.isalpha() for c in l)
        
        if not has_alpha or len(l) < 2:
            continue
        
        # Skip if it looks like address or contact info
        if phone_re.search(l) or '@' in l or website_re.search(l):
            continue
        
        # Skip if it's an address line
        line_lower = l.lower()
        if any(k in line_lower for k in address_keywords) or re.search(r'\d{5}', l):
            continue
        
        # Look for name: 1-4 words that could be a name
        if 1 <= len(words) <= 4:
            # Prefer all-caps lines (like "TUCKER") or mixed case lines with lowercase
            if l.isupper() or re.search(r"[a-z]", l):
                name = l
                break
        
        # Single uppercase word followed by another word might be last name + first name
        if len(words) >= 2 and words[0].isupper():
            name = l
            break
    
    # Second pass: look for clear designation matches
    for i, l in enumerate(cleaned_lines[:10]):
        if re.search(title_keywords, l, re.I):
            # Skip if this line contains address indicators (pincode, address keywords with word boundaries, commas, etc)
            line_lower = l.lower()
            # Use word boundaries for address keywords to avoid matching "state" in "Real Estate"
            has_address_keyword = any(re.search(r'\b' + k + r'\b', line_lower) for k in address_keywords if len(k) > 2)
            if has_address_keyword or re.search(r'\d{5}', l) or l.count(',') > 0:
                continue
            
            # Skip if this is just a state abbreviation (MD, CA, etc) in an address context
            if re.match(state_abbreviations, l.strip()):
                continue
            
            # Extract just the designation part, not the whole line
            match = re.search(title_keywords, l, re.I)
            if match:
                # Get the matched designation
                designation = match.group(0)
                # If it's part of a longer line, try to isolate it better
                if len(l) > len(designation) + 5:
                    # Check if this looks like "name designation" or "designation with extra text"
                    # If line contains address keywords, it's probably not the designation line
                    if not has_address_keyword:
                        designation = l
                break
    
    # Third pass: extract name from email if NOT found in text
    if not name and emails:
        for email in emails:
            # Extract local part (before @)
            local_part = email.split('@')[0]
            # Replace dots and underscores with spaces
            name_from_email = re.sub(r'[._]', ' ', local_part).title()
            # Clean up - remove numbers and special chars
            name_from_email = re.sub(r'\d+', '', name_from_email).strip()
            if len(name_from_email) >= 3:
                name = name_from_email
                break

    # Company detection: only extract if explicitly present in text with keywords
    for l in cleaned_lines:
        if any(k.lower() in l.lower() for k in ["pvt", "ltd", "private", "limited", "llp", "inc", "corporation", "company"]):
            company = l
            break

    # Address: improved grouping of consecutive address-like lines
    candidate_idxs = []
    for i, l in enumerate(lines):
        low = l.lower()
        # skip obvious non-address lines
        if phone_re.search(l) or (email_re.search(l) and '@' in l) or website_re.search(l):
            continue
        # Skip single-word lines unless they are address keywords
        word_count = len(l.split())
        if word_count == 1 and not any(k.lower() in low for k in address_keywords):
            continue
            
        # stricter pincode check: standalone 5-6 digit pincode
        has_pincode = re.search(r"\b\d{5,6}\b", l)
        has_keywords = any(k in low for k in address_keywords)
        has_commas = "," in l and len(l) > 12
        starts_with_number = re.match(r"^\d+\b", low)

        if has_pincode or has_keywords or has_commas or starts_with_number:
            candidate_idxs.append(i)

    # Group consecutive indices into address blocks
    address_blocks = []
    current_block = []
    last_idx = None
    for idx in candidate_idxs:
        if last_idx is None or idx - last_idx <= 1:
            current_block.append(lines[idx])
        else:
            if current_block:
                address_blocks.append(" ".join(current_block))
            current_block = [lines[idx]]
        last_idx = idx
    if current_block:
        address_blocks.append(" ".join(current_block))

    # pick the longest block as the address
    if address_blocks:
        address = max(address_blocks, key=lambda s: len(s))
    else:
        address = None

    # Detect languages
    langs = []
    try:
        from langdetect import detect_langs
        if raw_text:
            langs = [l.lang for l in detect_langs(raw_text)[:3]]
    except Exception:
        langs = ["eng"]

    return {
        "name": name,
        "designation": designation,
        "company": company,
        "mobile": phones,
        "email": emails,
        "address": address,
        "website": websites,
        "social": {},
        "extras": {},
        "language_detected": langs,
        "raw_text": raw_text,
    }


@app.post("/extract")
async def extract(file: UploadFile = File(...), languages: Optional[str] = Form("eng")):
    """Accepts multipart file upload; use the OpenAPI UI to upload a visiting card image."""
    image_bytes = await file.read()
    raw_text, ocr_conf, notes = ocr_from_bytes(image_bytes, languages.split("+"))
    parsed = parse_text(raw_text)
    
    # Calculate confidence based on fields extracted
    core_found = sum(1 for v in [parsed.get("name"), parsed.get("mobile"), parsed.get("email"), parsed.get("address")] if v)
    fields_score = core_found / 4.0
    
    # Combine OCR and parsing confidence
    final_conf = round(min(1.0, 0.7 * ocr_conf + 0.3 * fields_score), 2) if ocr_conf > 0 else fields_score
    parsed["confidence"] = final_conf
    
    if notes:
        parsed["notes"] = notes
    
    return parsed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
