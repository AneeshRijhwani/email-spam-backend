# from flask import Flask, request, jsonify
# import pickle
# import os

# app = Flask(__name__)

# # Load the models
# models = {
#     'NB': 'models/NB_model.pkl',
#     'DT': 'models/DT_model.pkl',
#     'LR': 'models/LR_model.pkl',
#     'RF': 'models/RF_model.pkl',
#     'Adaboost': 'models/Adaboost_model.pkl',
#     'BGc': 'models/Bgc_model.pkl',
#     'ETC': 'models/ETC_model.pkl',
#     'GBDT': 'models/GBDT_model.pkl',
#     'XGB': 'models/xgb_model.pkl'
# }

# def load_model(model_name):
#     if model_name in models:
#         with open(models[model_name], 'rb') as file:
#             return pickle.load(file)
#     else:
#         return None

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json(force=True)
    
#     email_text = data.get('email_text')
#     classifier = data.get('classifier')
#     print(email_text)
#     if not email_text or not classifier:
#         return jsonify({"error": "Please provide both 'email_text' and 'classifier'"}), 400
    
#     model = load_model(classifier)
    
#     if model is None:
#         return jsonify({"error": "Invalid classifier name"}), 400
    

#     with open('models/vectorizer.pkl', 'rb') as file:
#         vectorizer = pickle.load(file)
    
#     processed_text = vectorizer.transform([email_text])
    
#     prediction = model.predict(processed_text)
    
#     result = 'spam' if prediction[0] == 'spam' else 'not spam'
    
#     return jsonify({"prediction": result})

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the models
model_files = {
   'NB': 'models/NB_model.pkl',
    'DT': 'models/DT_model.pkl',
    'LR': 'models/LR_model.pkl',
    'RF': 'models/RF_model.pkl',
    'Adaboost': 'models/Adaboost_model.pkl',
    'BGc': 'models/Bgc_model.pkl',
    'ETC': 'models/ETC_model.pkl',
    'GBDT': 'models/GBDT_model.pkl',
    'XGB': 'models/xgb_model.pkl'
}

# Load the vectorizer
with open('./vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

def load_model(model_name):
    if model_name in model_files:
        with open(model_files[model_name], 'rb') as file:
            return pickle.load(file)
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    email_text = data.get('email_text')
    classifier = data.get('classifier')
    
    if not email_text or not classifier:
        return jsonify({"error": "Please provide both 'email_text' and 'classifier'"}), 400
    
    model = load_model(classifier)
    
    if model is None:
        return jsonify({"error": "Invalid classifier name"}), 400

    # Process the input text
    processed_text = vectorizer.transform([email_text])
    
    # Convert sparse matrix to dense if required
    if classifier in ['NB', 'DT', 'LR', 'RF', 'Adaboost', 'BGc', 'ETC', 'GBDT', 'XGB']:
        processed_text = processed_text.toarray()

    # Predict using the selected model
    prediction = model.predict(processed_text)
    
    # The model's prediction returns 0 or 1, corresponding to 'ham' or 'spam'
    result = 'spam' if prediction[0] == 1 else 'not spam'
    
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
