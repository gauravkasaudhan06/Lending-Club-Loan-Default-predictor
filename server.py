import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static')

# Load models and assets
models_dir = "models"
scaler_path = os.path.join(models_dir, "scaler.pkl")
columns_path = os.path.join(models_dir, "model_columns.pkl")
xgb_model_path = os.path.join(models_dir, "xgb_model.json")

# Placeholders for loaded components
scaler = None
model_columns = None
xgb_model = None

def load_inference_assets():
    global scaler, model_columns, xgb_model
    
    if not (os.path.exists(scaler_path) and os.path.exists(columns_path) and os.path.exists(xgb_model_path)):
        return False
        
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
        
    with open(columns_path, "rb") as f:
        model_columns = pickle.load(f)
        
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_model_path)
    return True

# Serve frontend
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

# Inference Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Make sure assets are loaded
    if xgb_model is None:
        success = load_inference_assets()
        if not success:
            return jsonify({
                "error": "Model files not found. Please run the training pipeline locally first using `python scripts/run_pipeline.py`."
            }), 500

    data = request.json
    if not data:
        return jsonify({"error": "No input data provided."}), 400

    try:
        # Initialize all feature columns to 0.0
        input_data = {col: 0.0 for col in model_columns}
        
        # Fill numeric features
        input_data['loan_amnt'] = float(data.get('loan_amnt', 10000))
        input_data['term'] = float(data.get('term', 36))
        input_data['int_rate'] = float(data.get('int_rate', 10.0))
        input_data['installment'] = float(data.get('installment', 300.0))
        input_data['annual_inc'] = float(data.get('annual_inc', 60000))
        input_data['dti'] = float(data.get('dti', 15.0))
        input_data['earliest_cr_line'] = float(data.get('earliest_cr_line', 2000))
        input_data['open_acc'] = float(data.get('open_acc', 10.0))
        input_data['pub_rec'] = float(data.get('pub_rec', 0.0))
        input_data['revol_bal'] = float(data.get('revol_bal', 10000.0))
        input_data['revol_util'] = float(data.get('revol_util', 50.0))
        input_data['total_acc'] = float(data.get('total_acc', 20.0))
        input_data['mort_acc'] = float(data.get('mort_acc', 0.0))
        input_data['pub_rec_bankruptcies'] = float(data.get('pub_rec_bankruptcies', 0.0))
        
        # Map categorical variables into dummy indicators
        categorical_mappings = {
            'sub_grade': data.get('sub_grade', 'B1'),
            'verification_status': data.get('verification_status', 'Not Verified'),
            'purpose': data.get('purpose', 'debt_consolidation'),
            'initial_list_status': data.get('initial_list_status', 'w'),
            'application_type': data.get('application_type', 'INDIVIDUAL'),
            'home_ownership': data.get('home_ownership', 'RENT'),
            'zip_code': data.get('zip_code', '30723')
        }
        
        for feature, value in categorical_mappings.items():
            col_name = f"{feature}_{value}"
            if col_name in input_data:
                input_data[col_name] = 1.0

        # Convert to DataFrame with exact column order
        df_input = pd.DataFrame([input_data])[model_columns]
        
        # Scale inputs
        scaled_input = scaler.transform(df_input)
        scaled_input = np.array(scaled_input).astype(np.float32)
        
        # Predict repayment probability (class 1 is Fully Paid, class 0 is Charged Off/Default)
        prob_fully_paid = float(xgb_model.predict_proba(scaled_input)[0, 1])
        prob_default = 1.0 - prob_fully_paid
        
        # Generate classification thresholds and recommendation
        risk_percentage = round(prob_default * 100, 2)
        
        if risk_percentage < 25:
            recommendation = "Approved (Low Risk)"
            risk_class = "low"
        elif risk_percentage < 55:
            recommendation = "Review Needed (Moderate Risk)"
            risk_class = "moderate"
        else:
            recommendation = "Denied (High Default Risk)"
            risk_class = "high"
            
        return jsonify({
            "probability_default": prob_default,
            "risk_percentage": risk_percentage,
            "recommendation": recommendation,
            "risk_class": risk_class
        })
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Initial load of models
    load_inference_assets()
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=True)
