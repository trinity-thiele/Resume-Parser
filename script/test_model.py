import pandas as pd

# Load the trained model
import sys
from pathlib import Path

# Add project root to path so imports work from any location
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.bag_of_words.bag_of_words_model import BagOfWords

project_root = Path(__file__).parent.parent
# Load model
model = BagOfWords.load_model(project_root / 'models/bag_of_words/bow_model.pkl')

# Single resume prediction
# Test with a new resume
with open(project_root / 'models/file_reading_application/processed_data/Fantigrossi-CV.txt', 'r') as file:
    new_resume = file.read()
# Clean resume using same cleaning function as used in data/clean/clean_datasets.py
print("Original Resume Text:")
print(new_resume)
def clean_resume(text):
    import re
    # Remove bullets (• or -) and asterisks
    text = re.sub(r'[•*-]', ' ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Strip leading/trailing spaces
    return text.strip()
new_resume = clean_resume(new_resume)
print("\nCleaned Resume Text:")
print(new_resume)
predicted_category = model.predict([new_resume])[0]
print(f"Predicted category: {predicted_category}")

# Get probability for each category
probabilities = model.predict_proba([new_resume])[0]
for idx, prob in enumerate(probabilities):
    print(f"Category {model.classifier.classes_[idx]}: {prob:.2%}")
    
# # Batch test predictions
# # Read test resumes from csv file
# test_df = pd.read_csv(project_root / 'data/clean/information-tech-set.csv')
# test_resumes = test_df['Resume'].tolist()
# predicted_categories = model.predict(test_resumes)
# test_df['Predicted_Category'] = predicted_categories
# # Save predictions to a new csv file
# test_df.to_csv(project_root / 'script/test_resumes_with_predictions.csv', index=False)

# # Print percentage of resumes predicted in each category
# category_counts = test_df['Predicted_Category'].value_counts(normalize=True) * 100
# for category, percentage in category_counts.items():
#     print(f"Category {category}: {percentage:.2f}%")
