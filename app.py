"""
Flask Web Application for Customer Churn Prediction
Provides a web interface for the ML pipeline
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load the trained pipeline
MODEL_PATH = 'models/best_churn_pipeline.pkl'
pipeline = None

def load_model():
    """Load the ML pipeline on startup"""
    global pipeline
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        print(f"✓ Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"⚠ Model not found at {MODEL_PATH}")

@app.route('/')
def home():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for making predictions
    Accepts JSON data and returns prediction with probability
    """
    try:
        # Get data from request
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Make prediction
        prediction = pipeline.predict(df)[0]
        probability = pipeline.predict_proba(df)[0, 1]
        
        # Determine risk level
        if probability > 0.7:
            risk_level = 'High'
            risk_color = '#e74c3c'
        elif probability > 0.4:
            risk_level = 'Medium'
            risk_color = '#f39c12'
        else:
            risk_level = 'Low'
            risk_color = '#2ecc71'
        
        # Prepare response
        response = {
            'success': True,
            'prediction': 'Churn' if prediction == 1 else 'No Churn',
            'probability': float(probability),
            'probability_percent': f"{probability * 100:.2f}%",
            'risk_level': risk_level,
            'risk_color': risk_color,
            'recommendation': get_recommendation(probability, data)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

def get_recommendation(probability, customer_data):
    """
    Generate personalized recommendations based on prediction
    """
    if probability > 0.7:
        return "🚨 High churn risk! Immediate action required. Consider offering special retention incentives or personal outreach."
    elif probability > 0.4:
        recommendations = []
        
        # Check contract type
        if customer_data.get('Contract') == 'Month-to-month':
            recommendations.append("Offer contract upgrade with discount")
        
        # Check services
        if customer_data.get('OnlineSecurity') == 'No':
            recommendations.append("Promote security services bundle")
        
        if customer_data.get('TechSupport') == 'No':
            recommendations.append("Offer complimentary tech support trial")
        
        # Check tenure
        tenure = int(customer_data.get('tenure', 0))
        if tenure < 12:
            recommendations.append("Provide new customer loyalty bonus")
        
        if recommendations:
            return "⚠️ Medium risk. Suggested actions: " + ", ".join(recommendations)
        else:
            return "⚠️ Medium risk. Monitor customer engagement and consider proactive outreach."
    else:
        return "✅ Low churn risk. Customer appears satisfied. Continue current service level."

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch predictions from CSV upload
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Remove customerID and Churn if present
        customer_ids = None
        if 'customerID' in df.columns:
            customer_ids = df['customerID'].tolist()
            df = df.drop('customerID', axis=1)
        
        if 'Churn' in df.columns:
            df = df.drop('Churn', axis=1)
        
        # Make predictions
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)[:, 1]
        
        # Prepare results
        results = []
        for i in range(len(predictions)):
            prob = probabilities[i]
            results.append({
                'customerID': customer_ids[i] if customer_ids else f"Customer_{i+1}",
                'prediction': 'Churn' if predictions[i] == 1 else 'No Churn',
                'probability': f"{prob * 100:.2f}%",
                'risk_level': 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
            })
        
        return jsonify({
            'success': True,
            'total_customers': len(results),
            'predicted_churns': sum(1 for r in results if r['prediction'] == 'Churn'),
            'results': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    load_model()
    print("\n" + "="*80)
    print("🚀 CUSTOMER CHURN PREDICTION WEB APP")
    print("="*80)
    print("\n✓ Server starting...")
    print("✓ Open your browser and go to: http://localhost:5000")
    print("\n" + "="*80 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
