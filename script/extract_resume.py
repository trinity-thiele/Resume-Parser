

import os
import json
import re
from typing import List, Dict, Any

import pdfplumber

try:
    import spacy
    NLP_MODEL = "en_core_web_sm"
    nlp = spacy.load(NLP_MODEL)
except Exception:
    # spaCy is optional in this version; we mainly use headings + regex.
    nlp = None


# ============================================
# PDF Text Extraction
# ============================================
def read_pdf(path: str) -> str:
    """Extract text from PDF using pdfplumber."""
    parts: List[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                parts.append(text)
    except Exception as e:
        print(f"[ERROR] Failed to read {path}: {e}")
        return ""

    # Normalize line endings
    return "\n".join(parts)


# ============================================
# Section Heading Helpers
# ============================================

# Patterns for each target section (what we WANT to extract)
SECTION_PATTERNS = {
    "skills": [
        r"key skills?",
        r"skills?"
    ],
    "education": [
        r"education"
    ],
    "experience": [
        r"professional experience",
        r"work experience",
        r"experience"
    ],
}

# Patterns for ANY heading (used to detect where a section ends)
# You can extend this if your resumes have more headings (Projects, Summary, etc.)
ALL_HEADING_PATTERNS = [
    r"key skills?",
    r"skills?",
    r"education",
    r"professional experience",
    r"work experience",
    r"experience",
    r"certifications?",
    r"projects?",
    r"summary",
    r"profile",
    r"professional summary",
]


def is_heading_line(line: str) -> bool:
    """Return True if the line looks like a section heading."""
    text = line.strip().lower()
    if not text:
        return False

    for pat in ALL_HEADING_PATTERNS:
        if re.fullmatch(pat, text, flags=re.IGNORECASE):
            return True

    return False


def match_section_name(line: str) -> str:
    """
    If 'line' is one of the target headings (skills, education, experience),
    return that section name. Otherwise return "".
    """
    text = line.strip().lower()
    for section_name, patterns in SECTION_PATTERNS.items():
        for pat in patterns:
            if re.fullmatch(pat, text, flags=re.IGNORECASE):
                return section_name
    return ""


# ============================================
# Section Extraction
# ============================================
def segment_sections_by_headings(text: str) -> Dict[str, List[str]]:
    """
    Split resume text by lines and extract raw lines for:
    - skills
    - education
    - experience

    Returns a dict:
        {
            "skills": [list of lines],
            "education": [list of lines],
            "experience": [list of lines]
        }
    """
    lines = [ln.rstrip() for ln in text.splitlines()]

    sections: Dict[str, List[str]] = {
        "skills": [],
        "education": [],
        "experience": [],
    }

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        section_name = match_section_name(line)

        if section_name:
            # We've hit a heading we care about.
            i += 1  # Move to first line AFTER the heading
            collected: List[str] = []

            while i < n:
                next_line = lines[i]
                # Stop if we hit another heading (any heading)
                if is_heading_line(next_line):
                    break
                collected.append(next_line)
                i += 1

            # Save non-empty lines
            cleaned = [ln for ln in collected if ln.strip()]
            sections[section_name].extend(cleaned)

        else:
            i += 1

    return sections


# ============================================
# Post-processing skills / education / experience
# ============================================
def extract_skills_from_lines(lines: List[str]) -> List[str]:
    """
    Convert the lines inside the "skills" section into a cleaner list.

    Heuristics:
    - Split on commas, pipes, bullets, and '•'.
    - Remove very short tokens and duplicate items.
    """
    raw_items: List[str] = []

    for ln in lines:
        # Split by common separators
        parts = re.split(r"[•\u2022\|\-·]|,", ln)
        for p in parts:
            item = p.strip()
            if len(item) > 1:
                raw_items.append(item)

    # Optional: use spaCy to drop obviously non-skill-ish sentences
    if nlp is not None:
        filtered: List[str] = []
        for it in raw_items:
            doc = nlp(it)
            # Heuristic: keep short noun-ish phrases, drop long sentences (with verbs)
            has_verb = any(tok.pos_ == "VERB" for tok in doc)
            if not has_verb or len(doc) <= 5:
                filtered.append(it)
        raw_items = filtered

    # Deduplicate while preserving order
    seen = set()
    skills: List[str] = []
    for item in raw_items:
        low = item.lower()
        if low not in seen:
            seen.add(low)
            skills.append(item)

    return skills


def normalize_block(lines: List[str]) -> str:
    """
    Join lines into a single block of text, removing extra blank lines.
    Used for Education and Experience.
    """
    # Remove leading/trailing empty lines
    cleaned = [ln.strip() for ln in lines if ln.strip()]
    return "\n".join(cleaned)


def extract_sections_from_text(text: str) -> Dict[str, Any]:
    """
    High-level function:
    - Takes raw resume text
    - Returns dict with ONLY skills, education, experience
    """
    raw_sections = segment_sections_by_headings(text)

    skills_list = extract_skills_from_lines(raw_sections.get("skills", []))
    education_block = normalize_block(raw_sections.get("education", []))
    experience_block = normalize_block(raw_sections.get("experience", []))

    return {
        "skills": skills_list,
        "education": education_block,
        "experience": experience_block,
    }


# ============================================
# Folder Processing
# ============================================
def process_resume_file(path: str) -> Dict[str, Any]:
    """Read one PDF and return extracted sections."""
    text = read_pdf(path)
    sections = extract_sections_from_text(text)
    sections["file_name"] = os.path.basename(path)  # helpful but not a field to extract
    return sections


def process_resume_folder(input_folder: str, output_json: str) -> None:
    """
    Walk through input_folder, find all PDFs, and write one JSON list:
    [
        {
            "file_name": "resume1.pdf",
            "skills": [...],
            "education": " ... ",
            "experience": " ... "
        },
        ...
    ]
    """
    results: List[Dict[str, Any]] = []

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(input_folder, fname)
        print(f"[INFO] Processing {fpath} ...")
        data = process_resume_file(fpath)
        results.append(data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Wrote {len(results)} resumes -> {output_json}")


# ============================================
# Main
# ============================================
if __name__ == "__main__":
    # Use the CURRENT folder where extract_resume.py + PDFs are
    INPUT_FOLDER = "."   # <--- IMPORTANT CHANGE
    OUTPUT_JSON = "resume_sections.json"

    print("[START] Running resume extraction...")
    print(f"[INFO] Looking for PDFs in: {os.path.abspath(INPUT_FOLDER)}")

    process_resume_folder(INPUT_FOLDER, OUTPUT_JSON)

    print(f"[DONE] Extraction finished. Results saved to {OUTPUT_JSON}")

