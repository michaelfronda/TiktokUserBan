import pickle
from flask import Flask, request, jsonify
import numpy as np 

# Load model and dv
with open("dv.pkl", "rb") as f_in: 
    dv = pickle.load(f_in)
    
with open("rf_model.pkl", "rb") as f_in: 
    model = pickle.load(f_in)

# instantiate
app = Flask('tt-user-ban')

# set path: /predict 
@app.route('/predict', methods=['POST']) # HTTP Request: Post 
def predict():
    # Get data 
    data = request.get_json()

    # Extract features
    X = dv.transform([data])
    
    # Make prediction 
    y_pred = model.predict_proba(X)[0, 1] 

    return jsonify({'prob': float(y_pred)}) # cast 

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

