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
with open(project_root / 'models/file_reading_application/processed_data/aws.txt', 'r') as file:
    new_resume = file.read()
predicted_category = model.predict([new_resume])[0]
print(f"Predicted category: {predicted_category}")

# Get probability for each category
probabilities = model.predict_proba([new_resume])[0]
for idx, prob in enumerate(probabilities):
    print(f"Category {model.classifier.classes_[idx]}: {prob:.2%}")