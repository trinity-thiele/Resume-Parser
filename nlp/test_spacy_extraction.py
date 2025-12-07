import re
import json
from pathlib import Path

import spacy
import docx          # for .docx files
import PyPDF2        # for .pdf files


# 1. Load spaCy English model
nlp = spacy.load("en_core_web_sm")


# ---------- STEP A: READING FILES (DOCX / PDF / TXT) ----------

def read_docx(path: Path) -> str:
    """Read text from a .docx file."""
    d = docx.Document(path)
    return "\n".join(p.text for p in d.paragraphs)


def read_pdf(path: Path) -> str:
    """Read text from a .pdf file (simple extraction)."""
    text_parts = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)


def read_txt(path: Path) -> str:
    """Read text from a .txt file."""
    return path.read_text(encoding="utf-8", errors="ignore")


def read_resume(path: Path) -> str:
    """
    Decide how to read the file based on its extension.
    This lets us use the SAME pipeline for docx, pdf, and txt.
    """
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx(path)
    elif suffix == ".pdf":
        return read_pdf(path)
    elif suffix in {".txt", ".text"}:
        return read_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# ---------- STEP B: SIMPLE FEATURE EXTRACTORS ----------

def extract_email(text: str):
    """Find the first email address using regex."""
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    return match.group(0) if match else None


def extract_phone(text: str):
    """Very rough phone-number detection."""
    pattern = r"(\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_name(doc):
    """
    Try to guess the candidate's name:
    - First PERSON entity from spaCy's NER,
    - If none, try the first line with 2+ title-case words.
    """
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()

    for line in doc.text.split("\n"):
        tokens = [t for t in line.split() if t.istitle()]
        if len(tokens) >= 2:
            return " ".join(tokens)
    return None


# a small example skill list for demo
SKILL_KEYWORDS = {
    "python", "java", "c++", "javascript", "react", "node.js",
    "sql", "pandas", "numpy", "machine learning", "deep learning",
    "nlp", "spacy", "linux", "git", "aws",
}

def extract_skills(doc):
    """
    Find skills by checking for keywords in the text.
    This is a simple keyword-based 'feature extraction'.
    """
    text = doc.text.lower()
    found = set()

    # phrase-based (e.g., "machine learning")
    for skill in SKILL_KEYWORDS:
        if " " in skill and skill in text:
            found.add(skill)

    # single-word skills
    tokens = {t.text.lower() for t in doc if not t.is_punct}
    for skill in SKILL_KEYWORDS:
        if " " not in skill and skill in tokens:
            found.add(skill)

    return sorted(found)


# ---------- STEP C: WRAP EVERYTHING INTO ONE FUNCTION ----------

def extract_features_from_file(path: Path):
    """
    Full pipeline:
    1) Read file (docx/pdf/txt) -> plain text
    2) Run spaCy nlp() to get tokens & entities
    3) Use regex + spaCy to extract simple features
    """
    raw_text = read_resume(path)
    doc = nlp(raw_text)

    features = {
        "file_name": path.name,
        "name": extract_name(doc),
        "email": extract_email(raw_text),
        "phone": extract_phone(raw_text),
        "skills": extract_skills(doc),
        # for now we just keep ALL named entities grouped by label
        "entities": {},
    }

    for ent in doc.ents:
        features["entities"].setdefault(ent.label_, set()).add(ent.text)

    # convert sets to lists so we can print as JSON
    for label in list(features["entities"].keys()):
        features["entities"][label] = sorted(features["entities"][label])

    return features


# ---------- STEP D: RUN ON YOUR SAMPLE DATASET ----------

if __name__ == "__main__":
    # Folder where you unzipped your resumes
    data_folder = Path("Resume_Sample")

    # You can filter here, e.g. only docx: data_folder.glob("*.docx")
    files = list(data_folder.glob("*"))

    if not files:
        print("No files found in", data_folder)
    else:
        for path in files:
            print("=" * 60)
            print("Processing:", path.name)
            feats = extract_features_from_file(path)
            print(json.dumps(feats, indent=2))
