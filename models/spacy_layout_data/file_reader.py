from pathlib import Path
import spacy
from spacy_layout import spaCyLayout
from spacy_layout.layout import TABLE_PLACEHOLDER

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

    # Extract tables
    tables_md = []
    for table_span in doc._.get(layout.attrs.doc_tables):
        # Get table if available
        df = getattr(table_span._, "data", None)
        if df is not None:
            # Convert table to markdown
            md = df.to_markdown(index=False)
            tables_md.append(md)
        else:
            # fallback to placeholder text
            tables_md.append(TABLE_PLACEHOLDER)

    return text, tables_md

def save_text_with_tables(text, tables, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n\n")
        if tables:
            f.write("Tables:\n\n")
            for i, table_md in enumerate(tables, 1):
                f.write(f"Table {i}:\n")
                f.write(table_md)
                f.write("\n\n")
    print(f"Text and tables saved to {output_path}")

def main():
    input_file = input("Enter path to PDF or DOCX file: ")
    output_file = input("Enter path for output text file: ")

    text_content, tables = read_document_with_tables(input_file)
    save_text_with_tables(text_content, tables, output_file)

if __name__ == "__main__":
    main()