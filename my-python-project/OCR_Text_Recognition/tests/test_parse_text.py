import sys
import pathlib
# ensure package root (parent of tests) is on sys.path so tests can import main_fixed
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from main_fixed import parse_text


def test_nehaal_card():
    raw = (
        "Nehaal Fakih\n+91-9004488330\n\n"
        "nehaal@91springboard.com\n\n"
        "9 SPRINGBOARD mum.andheri.lotus@91springboard.com\n\n"
        "cowork . network . grow\n\n"
        "www.91springboard.com\n\n"
    )
    parsed = parse_text(raw)
    assert parsed["name"] == "Nehaal Fakih"
    assert parsed["designation"] is None
    assert parsed["company"] and "SPRINGBOARD" in parsed["company"]
    assert "+919004488330" in parsed["mobile"]
    assert "http://www.91springboard.com" in parsed["website"]


def test_single_name_and_designation():
    raw = "Carol\nSenior Engineer\n"
    parsed = parse_text(raw)
    assert parsed["name"] == "Carol"
    assert parsed["designation"] == "Senior Engineer"


def test_email_company_inference():
    raw = "Alice Smith\nalice@acme.com\n"
    parsed = parse_text(raw)
    assert parsed["company"] == "ACME" or parsed["company"] == "ACME.COM"


def test_address_detection():
    raw = (
        "Plot No. D-5 Road No. 20, Marol MIDC, Andheri East\n"
        "Mumbai - 400069\n"
        "www.91springboard.com\n"
    )
    parsed = parse_text(raw)
    assert parsed["address"] is not None
    assert "Marol" in parsed["address"] or "Mumbai" in parsed["address"]