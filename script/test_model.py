import pandas as pd
import sys
from pathlib import Path
import os
import re
import numpy as np
import json

# Add project root to path so imports work from any location
sys.path.insert(0, str(Path(__file__).parent.parent))

# Assuming BagOfWords is correctly imported and available
from models.bag_of_words.bag_of_words_model import BagOfWords

project_root = Path(__file__).parent.parent
# Define the directory containing the processed text files
processed_data_dir = project_root / 'models' / 'file_reading_application' / 'processed_data'

# Load model
model = BagOfWords.load_model(project_root / 'models/bag_of_words/bow_model.pkl')

def clean_resume(text):
    # Remove bullets (• or -) and asterisks
    text = re.sub(r'[•*-]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Strip leading/trailing spaces
    return text.strip()

def main():
    print(f"\nStarting BOW classification on files in: {processed_data_dir}\n")

    # Get the list of all category classes from the trained model
    category_classes = model.classifier.classes_
    
    # Iterate over files in processed_data
    numFiles = 0
    results = []
    for filename in os.listdir(processed_data_dir):
        numFiles += 1
        if filename.endswith(".txt"):
            file_path = processed_data_dir / filename
            print(f"Reading {filename}...")

            with open(file_path, 'r', encoding='utf-8') as file:
                raw_resume = file.read()

            # Clean resume
            cleaned_resume = clean_resume(raw_resume)
            
            if not cleaned_resume:
                print(f"Skipping {filename}: Cleaned content is empty.")
                print("-" * 30)
                continue

            # Get probability for all categories
            probabilities = model.predict_proba([cleaned_resume])[0]
            
            # Combine categories and probabilities into a list of tuples
            category_probs = list(zip(category_classes, probabilities))
            
            # Sort the list by probability in descending order
            # item[1] is the probability percentage
            category_probs.sort(key=lambda item: item[1], reverse=True)

            # Get the top 3 categories
            top_3_categories = category_probs[:3]
            
            # Display the results in the desired format
            if top_3_categories:
                print(f"Highest category: {top_3_categories[0][0]} {top_3_categories[0][1]:.2%}")
            if len(top_3_categories) > 1:
                print(f"Second highest category: {top_3_categories[1][0]} {top_3_categories[1][1]:.2%}")
            if len(top_3_categories) > 2:
                print(f"Third highest category: {top_3_categories[2][0]} {top_3_categories[2][1]:.2%}")
            
            print("-" * 30) # Separator

            # Prepare top-3 structured data for persistence
            top3_struct = [
                {'category': str(cat), 'probability': float(prob)}
                for cat, prob in top_3_categories
            ]

            results.append({
                'file_name': filename,
                'bow_prediction': category_probs[0][0],
                'bow_confidence': float(category_probs[0][1]),
                'top_3_categories': top3_struct
            })
    print(f"\nProcessed {numFiles} files.")

    # After processing all files, save results (if any)
    if results:
        # Save full results (JSON) in the same directory as this script
        out_path = Path(__file__).parent / 'bow_results.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nSaved results to: {out_path}")

        # Build a compact CSV summary (one row per resume) with BOW results only
        summary_rows = []
        for item in results:
            row = {
                'file_name': item.get('file_name'),
                'bow_prediction': item.get('bow_prediction'),
                'bow_confidence': item.get('bow_confidence'),
                'bow_top3': '|'.join([t.get('category') for t in item.get('top_3_categories', [])])
            }
            summary_rows.append(row)

        csv_path = Path(__file__).parent / 'bow_predictions.csv'
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"Saved CSV summary to: {csv_path}")

if __name__ == '__main__':
    main()

# The original batch test predictions are removed for clarity