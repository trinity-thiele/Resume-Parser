import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import re
from pathlib import Path

"""
Bag of Words Model for Resume Classification
"""

"""
A class to handle Bag of Words vectorization and classification for resumes
"""
class BagOfWords:
    """
    Initialize the Bag of Words model
        
    Parameters:
    - max_features: Maximum number of features (words) to keep
    - min_df: Minimum document frequency (ignore rare words): 2 - ignore any token that appears in fewer than 2 documents
    - max_df: Maximum document frequency (ignore too common words): 0.8 - ignore any token that appears in more than 80% of documents
    - ngram_range: Range of n-grams to consider: (1,2): - consider single words and word pairs as features
    - stop_words: Remove stop words from resulting tokens
    - lowercase: Convert all characters to lowercase before tokenizing
    """
    def __init__(self, max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2)):
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial'
        )
        self.is_fitted = False
    
    """
    Fit the vectorizer and classifier on training data
        
    Parameters:
    - X_train: List of resume texts
    - y_train: List of labels (categories or clusters)
    """
    def fit(self, X_train, y_train):
        
        # Fit vectorizer and transform training data
        X_train_vectors = self.vectorizer.fit_transform(X_train)
        
        # Train classifier
        self.classifier.fit(X_train_vectors, y_train)
        
        self.is_fitted = True
        
        return self
    
    """
    Transform text data to bag of words vectors
    """
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        return self.vectorizer.transform(X)
    
    """
    Predict categories for new resumes
    """
    def predict(self, X):
        X_vectors = self.transform(X)
        return self.classifier.predict(X_vectors)
    
    """
    Get probability predictions for each class
    """
    def predict_proba(self, X):
        X_vectors = self.transform(X)
        return self.classifier.predict_proba(X_vectors)
    
    """
    Test model accuracy using testing data
    """
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        conf_matrix = confusion_matrix(y_test, predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }
    
    """
    Get the most important features for each cluster
    Use for LLM-based feedback
    """    
    def get_top_features_per_cluster(self, n_features=20):
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get coefficients for each class
        top_features = {}
        for idx, class_label in enumerate(self.classifier.classes_):
            # Get coefficients for this class
            coef = self.classifier.coef_[idx]
            
            # Get indices of top features
            top_indices = np.argsort(coef)[-n_features:][::-1]
            
            # Get feature names and their coefficients
            top_words = [(feature_names[i], coef[i]) for i in top_indices]
            
            top_features[class_label] = top_words
        
        return top_features
    
    """
    Save trained model
    """
    def save_model(self, filepath):
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    """
    Load saved (trained) model
    """    
    @classmethod
    def load_model(cls, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance
        instance = cls()
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


"""
Example training script for the resume classifier
"""
def train_bow_model():
    # Load clustered data
    df = pd.read_csv('data/clustered/clustered_resumes_1.csv')
    
    # Prepare data using cluster names as labels
    X = df['Text'].values
    y = df['cluster_name'].values 
    
    # Split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train model
    print("Training Bag of Words model...")
    bow_model = BagOfWords(max_features=5000, ngram_range=(1, 2))
    # Train model on training data
    bow_model.fit(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    results = bow_model.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    # Get top features per cluster
    print("\nTop features per cluster:")
    top_features = bow_model.get_top_features_per_cluster(n_features=10)
    for category, features in top_features.items():
        print(f"\n{category}:")
        for word, score in features[:5]:
            print(f"  {word}: {score:.4f}")
    
    # Save model - ensure path exists
    Path('models/bag_of_words').mkdir(parents=True, exist_ok=True)
    """
    Save to pickle file - commonly used for storing machine learning models
    Allows Python objects to be serialized (converted) into byte stream and deserialized back into Python object given byte stream
    """
    bow_model.save_model('models/bag_of_words/bow_model.pkl')
    print("\nModel saved to models/bag_of_words/bow_model.pkl")
    
    return bow_model

if __name__ == "__main__":
    model = train_bow_model()