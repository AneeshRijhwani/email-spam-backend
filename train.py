
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Data cleaning and processing functions
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

def load_and_process_data(file_path):
    # Load dataset with specified encoding
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='latin1')  # Try a different encoding

    # Data cleaning
    data = data.dropna(subset=['v1', 'v2'])

    # Extract features and labels
    X = data['v2']
    y = data['v1']


    # Feature extraction
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X).toarray()

    # Encoding labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    return X, y, vectorizer, label_encoder

# Load and process data
X, y, vectorizer, label_encoder = load_and_process_data('./spam.csv')


with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize models
models = {
    'NB': GaussianNB(),
    'DT': DecisionTreeClassifier(),
    'LR': LogisticRegression(),
    'RF': RandomForestClassifier(),
    'Adaboost': AdaBoostClassifier(),
    'Bgc': GradientBoostingClassifier(),
    'ETC': ExtraTreesClassifier(),
    'GBDT': GradientBoostingClassifier(),
    'xgb': GradientBoostingClassifier()
}


# Train models and save learnings
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f'Model: {name}')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    # Save model
    with open(f'models/{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)


print("Training complete and models saved.")
