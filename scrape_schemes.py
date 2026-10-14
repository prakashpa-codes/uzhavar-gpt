"""One-time data refresh: pull Tamil Nadu agriculture schemes from official portals.

Sources
  1. TNAGRISNET (tnagrisnet.tn.gov.in/home/schemes) -- bilingual EN/TA scheme master
     for Agriculture, Horticulture, Agri Engineering, Marketing, Sugarcane, etc.
  2. TNAGRISNET GoScheme (people_app/GoScheme) -- component-wise subsidy rates + GO links.
  3. tn.gov.in scheme list for the Agriculture - Farmers Welfare Department.

Writes markdown files into source-data/, which ingest.py then indexes.
The Streamlit app never touches the web -- run this only when refreshing data.

    python scrape_schemes.py
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

OUT_DIR = Path("source-data")
HEADERS = {"User-Agent": "Mozilla/5.0 (UzhavarGPT data refresh)"}

# ---------------------------------------------------------------- TNAGRISNET
AGRISNET_API = "https://www.tnagrisnet.tn.gov.in/Scheme_master/getSchemes/"
AGRISNET_DEPTS = {
    "A": "Agriculture / வேளாண்மை துறை",
    "H": "Horticulture / தோட்டக்கலை துறை",
    "E": "Agricultural Engineering / வேளாண் பொறியியல் துறை",
    "SO": "Seed Certification & Organic Certification / விதை சான்றளிப்பு மற்றும் அங்கக சான்றளிப்பு துறை",
    "M": "Marketing & Agri Business / வேளாண் விற்பனை மற்றும் வேளாண் வணிகத்துறை",
    "S": "Sugarcane / சர்க்கரைத் துறை",
    "T": "Tamil Nadu Watershed Development Agency / தமிழ்நாடு நீர்வடிப்பகுதி மேம்பாட்டு முகமை",
    "TU": "Tamil Nadu Agricultural University / தமிழ் நாடு வேளாண்மை பல்கலைக்கழகம்",
}

GOSCHEME_PAGE = "https://www.tnagrisnet.tn.gov.in/people_app/GoScheme"
GOSCHEME_LIST = "https://www.tnagrisnet.tn.gov.in/people_app/goScheme/pending_ajax_list"
GOSCHEME_DETAIL = "https://www.tnagrisnet.tn.gov.in/people_app/goScheme/getSchemeDet/"

# --------------------------------------------------------------- tn.gov.in
TNGOV_BASE = "https://www.tn.gov.in/"
TNGOV_LIST = TNGOV_BASE + "scheme_list.php?dep_id=Mg=="  # Agriculture - Farmers Welfare Dept
TNGOV_LABELS = [
    "Concerned Department:", "Concerned District:", "Organisation Name:", "Scheme Details:",
    "Scheme Title/Name:", "Associated Scheme:", "Sponsered By:", "Funding Pattern:",
    "Beneficiaries:", "Types of Benefits:", "Eligibility criteria:", "Income:", "Age From:",
    "Age To:", "Community:", "How To avail:", "Validity of the Scheme:", "Introduced On:",
    "Description:", "Scheme Type:", "Uploaded File:",
]
TNGOV_SKIP_LABELS = {"Concerned District", "Organisation Name", "Scheme Details",
                     "Eligibility criteria", "Scheme Type", "Uploaded File"}


def clean(html_text: str) -> str:
    """Strip the inline <small>/<br> tags the portals embed in field values."""
    text = BeautifulSoup(html_text or "", "html.parser").get_text("\n")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def scrape_agrisnet() -> int:
    sections = []
    for code, dept in AGRISNET_DEPTS.items():
        resp = requests.post(AGRISNET_API + code, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        for s in resp.json():
            if s.get("delete_status") == "yes":
                continue
            rows = [
                f"## {s['Scheme_Name'].strip()} / {s['Scheme_Name_Tamil'].strip()}",
                "Source: TNAGRISNET, Department of Agriculture - Farmers Welfare, Govt. of Tamil Nadu "
                "(https://www.tnagrisnet.tn.gov.in/home/schemes/)",
                f"Department: {dept}",
                f"Level: Tamil Nadu State implemented scheme ({s.get('year') or 'year n/a'})",
            ]
            for label, en_key, ta_key in [
                ("Scheme Description / திட்டத்தின் விளக்கம்", "scheme_desc", "scheme_desc_tamil"),
                ("Eligibility / தகுதி", "eligibility", "eligibility_tamil"),
                ("Documents Required / தேவையான ஆவணங்கள்", "doc_req", "doc_req_tamil"),
            ]:
                en, ta = clean(s.get(en_key)), clean(s.get(ta_key))
                if en or ta:
                    rows.append(f"\n### {label}\n{en}\n{ta if ta not in en else ''}".rstrip())
            if len(rows) > 4:  # skip schemes with no content at all
                sections.append("\n".join(rows))
        print(f"  TNAGRISNET {code}: total so far {len(sections)}")

    header = "# Tamil Nadu Agriculture Schemes - TNAGRISNET (bilingual English / Tamil)\n"
    (OUT_DIR / "tnagrisnet_schemes.md").write_text(header + "\n\n".join(sections) + "\n", encoding="utf-8")
    return len(sections)


def scrape_tngov() -> int:
    def soup_of(url):
        page = requests.get(url, headers=HEADERS, timeout=30)
        page.raise_for_status()
        return BeautifulSoup(page.text, "html.parser")

    links = []
    for a in soup_of(TNGOV_LIST).find_all("a", href=True):
        if a["href"].startswith("scheme_details.php") and a["href"] not in links:
            links.append(a["href"])

    sections = []
    for href in links:
        url = TNGOV_BASE + href
        try:
            lines = soup_of(url).get_text("\n", strip=True).splitlines()
        except requests.RequestException as e:
            print(f"  skip {href}: {e}")
            continue
        # Flat 'Label:' / value lines; walk them from the first label to the footer.
        data, current = {}, None
        start = next((i for i, l in enumerate(lines) if l == "Concerned Department:"), len(lines))
        for line in lines[start:]:
            if line.startswith("Content owned and Maintained by"):
                break
            if line in TNGOV_LABELS:
                current = line.rstrip(":")
                data[current] = ""
            elif current:
                data[current] = (data[current] + " " + line).strip()

        title = data.get("Scheme Title/Name", "")
        # This listing also carries MSME/industry schemes; keep only farmer-facing ones.
        blob = " ".join([title, data.get("Beneficiaries", ""), data.get("Description", "")]).lower()
        if not title or "farm" not in blob:
            print(f"  skip (not farmer-facing): {title}")
            continue
        rows = [f"## {title}", f"Source: {url}", "Level: Tamil Nadu State scheme (tn.gov.in)"]
        for key, value in data.items():
            if key not in TNGOV_SKIP_LABELS and key != "Scheme Title/Name" and value and value.lower() not in {"-", "na"}:
                rows.append(f"- **{key}:** {value}")
        sections.append("\n".join(rows))
        time.sleep(0.3)  # be polite to the government server

    header = f"# Tamil Nadu Agriculture - Farmers Welfare Department Schemes (tn.gov.in)\nScraped from {TNGOV_LIST}\n"
    (OUT_DIR / "tn_gov_agri_schemes.md").write_text(header + "\n\n".join(sections) + "\n", encoding="utf-8")
    return len(sections)


def scrape_goscheme() -> int:
    """TNAGRISNET people_app GoScheme: component-wise subsidy patterns + GO links."""
    sections = []
    for code, dept in AGRISNET_DEPTS.items():
        resp = requests.post(GOSCHEME_LIST, headers=HEADERS, timeout=30,
                             data={"draw": 1, "start": 0, "length": 500, "dept": code, "scheme": ""})
        resp.raise_for_status()
        for row in resp.json().get("data", []):
            _, _, name_html, desc, eligibility, docs, go_html = row
            name_tag = BeautifulSoup(name_html, "html.parser").find("a")
            scheme_id = re.search(r"getSchemeDet\((\d+)\)", name_html)
            go_link = BeautifulSoup(go_html, "html.parser").find("a")
            rows = [
                f"## {name_tag.get_text(strip=True) if name_tag else clean(name_html)}",
                f"Source: TNAGRISNET GO & Scheme list ({GOSCHEME_PAGE})",
                f"Department: {dept}",
                f"- **Scheme Description:** {clean(desc)}",
                f"- **Eligibility:** {clean(eligibility)}",
                f"- **Documents Required:** {clean(docs)}",
            ]
            if go_link and go_link.get("href"):
                rows.append(f"- **Government Order / Guidelines:** {go_link['href']}")
            if scheme_id:
                body = requests.post(GOSCHEME_DETAIL + scheme_id.group(1), headers=HEADERS, timeout=30).text.strip()
                comps = json.loads(body) if body.startswith("[") else []  # empty body = no components
                if comps:
                    rows.append("\n### Component-wise subsidy / இனங்கள் வாரியான மானியம்")
                for c in comps:
                    en = clean(c.get("pattern_of_subsidy")).replace("\n", " ")
                    ta = clean(c.get("pattern_of_subsidy_tamil")).replace("\n", " ")
                    if not (en or ta):
                        continue
                    rows.append(
                        f"- **{(c.get('input_type_name') or '').strip()} - {(c.get('class_Name') or '').strip()}"
                        f" / {(c.get('class_name_Tamil') or '').strip()}:** {en} | {ta}"
                    )
                time.sleep(0.3)
            sections.append("\n".join(rows))
        print(f"  GoScheme {code}: total so far {len(sections)}")

    header = "# Tamil Nadu Agriculture Schemes - Component-wise Subsidy Patterns (TNAGRISNET GoScheme)\n"
    (OUT_DIR / "tnagrisnet_goscheme_subsidies.md").write_text(header + "\n\n".join(sections) + "\n", encoding="utf-8")
    return len(sections)


if __name__ == "__main__":
    print(f"TNAGRISNET: saved {scrape_agrisnet()} schemes")
    print(f"GoScheme  : saved {scrape_goscheme()} schemes")
    print(f"tn.gov.in : saved {scrape_tngov()} schemes")
