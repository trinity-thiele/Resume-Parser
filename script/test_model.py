# Load the trained model
from models.bag_of_words.bag_of_words_model import BagOfWords

model = BagOfWords.load_model('models/bag_of_words/bow_model.pkl')

# Single resume prediction
new_resume = ""
predicted_category = model.predict([new_resume])[0]
print(f"Predicted category: {predicted_category}")

# Get probability for each category
probabilities = model.predict_proba([new_resume])[0]
for idx, prob in enumerate(probabilities):
    print(f"Category {model.classifier.classes_[idx]}: {prob:.2%}")