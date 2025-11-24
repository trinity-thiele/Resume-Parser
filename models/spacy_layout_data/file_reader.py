from pathlib import Path
import spacy
from spacy_layout import spaCyLayout
import os

def read_document_with_tables(file_path, separator="\n\n"):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Create blank spaCy pipeline
    nlp = spacy.blank("en")

    # Initialize spaCyLayout
    layout = spaCyLayout(nlp, separator=separator)

    # Load the document
    doc = layout(file_path)

    # Combine span texts into main text
    span_group = layout.attrs.span_group
    text = separator.join([span.text for span in doc.spans[span_group]])

    return text

def save_text(text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def main():
    for filename in os.listdir("./"):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join("./", filename)
            text= read_document_with_tables(pdf_path)
            save_text(text, (filename[:-3] + "txt"))

if __name__ == "__main__":
    main()