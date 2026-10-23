"""Guards for the parsing in ingest.py. Run with:  pytest

These run offline (no OpenAI calls), so they're quick to run after every
scrape_schemes.py refresh or a new PDF, before paying to rebuild the index.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingest import is_clean_line, load_markdown  # noqa: E402


# ---- is_clean_line: what survives from the ICAR PDFs ----

def test_keeps_plain_english():
    assert is_clean_line("Subsidy of Rs.15,000 per hectare for drip irrigation (PMKSY).")


def test_drops_tamil_script():
    # Clean Tamil comes from the portal markdown; Tamil in PDFs is dropped on purpose.
    assert not is_clean_line("பிரதம மந்திரி கிசான் திட்டம்")
    assert not is_clean_line("PM-KISAN / பிரதம மந்திரி கிசான்")


def test_drops_legacy_font_mojibake():
    # This is how the legacy Tamil font comes out of pypdf.
    assert not is_clean_line("Ö¼ïòÂ öÁÎõ¸ öÁ¼ Ã¸ªÁõ")


def test_drops_replacement_char():
    assert not is_clean_line("Rs.5000 � per acre")


def test_allows_an_odd_accent_in_long_english_line():
    # One stray accented letter (a name, "café") shouldn't throw away a useful line.
    line = "Assistance is given to farmer producer groups for setting up a café style outlet near the Uzhavar Sandhai market."
    assert is_clean_line(line)


# ---- load_markdown: one Document per "## scheme" section ----

SAMPLE = """# Tamil Nadu Agriculture Schemes - TNAGRISNET
Scraped from somewhere. This header is not a scheme.

## National Horticulture Mission (NHM) / தேசிய தோட்டக்கலை இயக்கம்
Source: TNAGRISNET
Department: Horticulture / தோட்டக்கலை துறை

### Eligibility / தகுதி
All farmers.

## Training to Farmers
Source: https://www.tn.gov.in/scheme_details.php?id=1
- **Concerned Department:** Agriculture - Farmers Welfare Department
- **Funding Pattern:** Rs.300 per farmer for 2 days.

## Scheme without a department line
Some text.
"""


def load_sample(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text(SAMPLE, encoding="utf-8")
    return load_markdown(path)


def test_one_document_per_scheme_and_header_skipped(tmp_path):
    docs = load_sample(tmp_path)
    assert [d.metadata["scheme"] for d in docs] == [
        "National Horticulture Mission (NHM) / தேசிய தோட்டக்கலை இயக்கம்",
        "Training to Farmers",
        "Scheme without a department line",
    ]


def test_sections_never_mix(tmp_path):
    nhm, training, _ = load_sample(tmp_path)
    # "###" sub-headings stay inside their scheme, the next "##" starts a new one.
    assert "All farmers." in nhm.page_content
    assert "Rs.300" not in nhm.page_content
    assert training.page_content.startswith("## Training to Farmers")
    assert "Horticulture" not in training.page_content


def test_metadata_source_and_department(tmp_path):
    nhm, training, plain = load_sample(tmp_path)
    assert nhm.metadata["source"] == "sample.md"
    assert nhm.metadata["department"] == "Horticulture"  # English part only
    assert training.metadata["department"] == "Agriculture - Farmers Welfare Department"
    assert "department" not in plain.metadata


def test_real_scraped_files_still_parse():
    # Catches a scraper change that silently breaks the "## " format.
    data = Path(__file__).resolve().parent.parent / "source-data"
    for name in ["tnagrisnet_schemes.md", "tnagrisnet_goscheme_subsidies.md", "tn_gov_agri_schemes.md"]:
        docs = load_markdown(data / name)
        assert len(docs) >= 20, name
        assert all(d.page_content.startswith("## ") for d in docs)
        assert all("department" in d.metadata for d in docs), name
