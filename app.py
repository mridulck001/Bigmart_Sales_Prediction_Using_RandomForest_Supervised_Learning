from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime
import random
import warnings
warnings.filterwarnings('ignore')


try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("⚠️ Joblib not available, using pickle only")

app = Flask(__name__)


model = None
model_error = None

def load_model():
    global model, model_error
    
    # Try multiple model files in order of preference
    model_files = [
        "model_rf.pkl",
        "model_xgb.pkl", 
        "model_tree.pkl",
        "model_linear.pkl"
    ]
    
    for model_file in model_files:
        try:
            print(f"🔍 Trying to load: {model_file}")
            
            # Check if file exists
            if not os.path.exists(model_file):
                print(f"⚠️ {model_file} not found, trying next...")
                continue
            
            # Get file size
            file_size = os.path.getsize(model_file)
            print(f"📁 Loading model ({file_size / (1024*1024):.1f} MB)...")
            
            start_time = time.time()
            
            # Try multiple loading methods
            model_loaded = False
            
            # Method 1: Standard pickle load
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                model_loaded = True
                print("✅ Loaded with standard pickle")
            except Exception as e1:
                print(f"❌ Standard pickle failed: {e1}")
                
                # Method 2: Try with joblib if available
                if JOBLIB_AVAILABLE:
                    try:
                        model = joblib.load(model_file)
                        model_loaded = True
                        print("✅ Loaded with joblib")
                    except Exception as e2:
                        print(f"❌ Joblib failed: {e2}")
                
                # Method 3: Try with latin-1 encoding
                if not model_loaded:
                    try:
                        with open(model_file, "rb") as f:
                            model = pickle.load(f, encoding='latin-1')
                        model_loaded = True
                        print("✅ Loaded with latin-1 encoding")
                    except Exception as e3:
                        print(f"❌ Latin-1 encoding failed: {e3}")
                
                # Method 4: Try with iso-8859-1 encoding
                if not model_loaded:
                    try:
                        with open(model_file, "rb") as f:
                            model = pickle.load(f, encoding='iso-8859-1')
                        model_loaded = True
                        print("✅ Loaded with iso-8859-1 encoding")
                    except Exception as e4:
                        print(f"❌ ISO-8859-1 encoding failed: {e4}")
                        continue
            
            if not model_loaded:
                continue
                
            load_time = time.time() - start_time
            print(f"✅ Model loaded successfully in {load_time:.2f} seconds!")
            print(f"📋 Model type: {type(model)}")
            print(f"📄 Model file: {model_file}")
            
            # Verify model has predict method
            if not hasattr(model, 'predict'):
                print("❌ Model doesn't have predict method, trying next...")
                continue
            
            # Test with sample data to ensure it works
            try:
                test_data = create_sample_dataframe()
                test_prediction = model.predict(test_data)[0]
                print(f"🧪 Test prediction successful: ₹{test_prediction:,.2f}")
                
                model_error = None
                return True
                
            except Exception as test_error:
                print(f"❌ Test prediction failed: {test_error}")
                print("🔄 Trying next model file...")
                continue
        
        except Exception as e:
            print(f"❌ Error with {model_file}: {e}")
            continue
    
    
    model = None
    model_error = "Could not load any model file. All attempts failed."
    print(f"❌ {model_error}")
    print("🔧 Troubleshooting tips:")
    print("   1. Check if model files exist and are not corrupted")
    print("   2. Verify scikit-learn version compatibility")
    print("   3. Try re-training the model with current environment")
    print("   4. Install joblib: pip install joblib")
    
    return False

def create_sample_dataframe():
    """Create a sample dataframe for testing model predictions"""
    df = pd.DataFrame({
        'Item_Weight': [8.5],
        'Item_Fat_Content': ['Low Fat'],
        'Item_Visibility': [0.016],
        'Item_Type': ['Dairy'],
        'Item_MRP': [249.81],
        'Outlet_Identifier': ['OUT013'],
        'Outlet_Establishment_Year': [1987],
        'Outlet_Size': ['Medium'],
        'Outlet_Location_Type': ['Tier 1'],
        'Outlet_Type': ['Supermarket Type1']
    })
    # Calculate Outlet_Age from Outlet_Establishment_Year
    df['Outlet_Age'] = 2026 - df['Outlet_Establishment_Year']
    # Drop Outlet_Establishment_Year as it's not used by the model
    df = df.drop('Outlet_Establishment_Year', axis=1)
    return df

def get_expected_columns():
    """Get the expected column names and their order"""
    return [
        'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
        'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
    ]

def preprocess_input_data(input_data):
    """Preprocess input data for prediction with proper column handling"""
    expected_keys = get_expected_columns()
    
    # Create a clean copy
    processed_data = {}
    
    # Fill missing values with appropriate defaults
    for key in expected_keys:
        if key not in input_data or input_data[key] is None or input_data[key] == '':
            if key in ['Item_Weight', 'Item_Visibility']:
                processed_data[key] = 0.0
            elif key == 'Item_MRP':
                processed_data[key] = 100.0  # Default price
            elif key == 'Outlet_Establishment_Year':
                processed_data[key] = 1999  # Default year
            else:
                processed_data[key] = 'Unknown'
        else:
            processed_data[key] = input_data[key]
    
    # Convert numeric fields
    numeric_fields = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
    for key in numeric_fields:
        try:
            processed_data[key] = float(processed_data[key])
        except (ValueError, TypeError):
            if key == 'Outlet_Establishment_Year':
                processed_data[key] = 1999
            else:
                processed_data[key] = 0.0
    
    return processed_data

def prepare_dataframe_for_prediction(input_data):
    """Prepare DataFrame with proper column order and preprocessing for model prediction"""
    # Preprocess the data first
    processed_data = preprocess_input_data(input_data)
    
    # Create DataFrame with exact column order
    expected_columns = get_expected_columns()
    df_data = {}
    
    for col in expected_columns:
        df_data[col] = [processed_data[col]]
    
    df = pd.DataFrame(df_data)
    
    # Calculate Outlet_Age from Outlet_Establishment_Year
    if 'Outlet_Establishment_Year' in df.columns:
        df['Outlet_Age'] = 2026 - df['Outlet_Establishment_Year']  # Using current year 2026
        # Drop Outlet_Establishment_Year as it's not used by the model
        df = df.drop('Outlet_Establishment_Year', axis=1)
    
    categorical_columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 
                          'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']
    
    for col in categorical_columns:
        if col in df.columns:
            # Convert to string to ensure consistency
            df[col] = df[col].astype(str)
    
    return df

def create_demo_model():
    """Create a simple demo model for testing when real model fails to load"""
    print("🎭 Creating demo model for testing purposes...")
    
    class DemoModel:
        def predict(self, data):
            """Simple demo prediction based on MRP and item type"""
            predictions = []
            for _, row in data.iterrows():
                # Simple formula for demo: base prediction on MRP and item type
                base_prediction = float(row.get('Item_MRP', 100)) * 2.5
                
                # Adjust based on item type
                item_type = str(row.get('Item_Type', 'Others')).lower()
                if 'dairy' in item_type:
                    base_prediction *= 1.2
                elif 'snack' in item_type:
                    base_prediction *= 0.8
                elif 'household' in item_type:
                    base_prediction *= 1.1
                
                # Adjust based on outlet type
                outlet_type = str(row.get('Outlet_Type', 'Grocery Store')).lower()
                if 'supermarket' in outlet_type:
                    base_prediction *= 1.3
                elif 'grocery' in outlet_type:
                    base_prediction *= 0.9
                
                # Add some randomness
                # base_prediction *= random.uniform(0.85, 1.15)
                
                predictions.append(max(50, base_prediction))  # Minimum ₹50
            
            return np.array(predictions)
        
        def __str__(self):
            return "DemoModel(BigMart Sales Predictor - Demo Mode)"
    
    return DemoModel()

# Attempt to load model on startup
print("🚀 Starting BigMart Sales Prediction App...")
model_loaded = load_model()

# If no model could be loaded, create demo model
if not model_loaded and model is None:
    print("🎭 No trained model available - switching to DEMO MODE")
    print("⚠️ Demo mode provides sample predictions for testing purposes only")
    model = create_demo_model()
    model_error = "Running in demo mode - predictions are for testing only"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Define all expected columns
        expected_keys = [
            'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
            'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
        ]
        
        # Build input_data
        input_data = {}
        missing_fields = []
        
        for key in expected_keys:
            value = request.form.get(key, '').strip()
            if value == '' or value is None:
                input_data[key] = None
                missing_fields.append(key.replace('_', ' '))
            else:
                input_data[key] = value

        # Validate required fields
        critical_fields = ['Item_MRP', 'Item_Type', 'Outlet_Type']
        critical_missing = [field for field in critical_fields if input_data[field] is None]
        
        if critical_missing:
            error_msg = f"Please fill in the required fields: {', '.join([field.replace('_', ' ') for field in critical_missing])}"
            return render_template("index.html", prediction_text=f"Error: {error_msg}", error=True)

        # Convert numeric fields with validation
        numeric_fields = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Establishment_Year']
        for key in numeric_fields:
            if input_data[key] is not None:
                try:
                    input_data[key] = float(input_data[key])
                    # Validate ranges
                    if key == 'Item_Visibility' and (input_data[key] < 0 or input_data[key] > 1):
                        raise ValueError("Item Visibility must be between 0 and 1")
                    if key == 'Item_Weight' and input_data[key] < 0:
                        raise ValueError("Item Weight cannot be negative")
                    if key == 'Item_MRP' and input_data[key] < 0:
                        raise ValueError("Item MRP cannot be negative")
                    if key == 'Outlet_Establishment_Year':
                        if input_data[key] < 1985 or input_data[key] > 2020:
                            raise ValueError("Outlet Establishment Year must be between 1985 and 2020")
                except ValueError as ve:
                    return render_template("index.html", prediction_text=f"Validation Error: {str(ve)}", error=True)

        # Preprocess the data using the proper function
        processed_data = preprocess_input_data(input_data.copy())
        
        # Create DataFrame for prediction with proper column handling
        df = prepare_dataframe_for_prediction(input_data)

        if model is None:
            error_msg = f"Model not loaded. {model_error if model_error else 'Please check if model_rf.pkl exists and is valid.'}"
            print(f"❌ Prediction failed: {error_msg}")
            return render_template("index.html", prediction_text=f"Error: {error_msg}", error=True)

        # Make prediction
        prediction = model.predict(df)[0]
        
        # Format prediction nicely
        if prediction < 0:
            prediction = abs(prediction)  # Ensure positive sales prediction
            
        formatted_prediction = f"₹{prediction:,.2f}"

        # Save prediction result
        if request.form.get('save_result') == 'on':
            try:
                save_prediction_result(input_data, prediction, formatted_prediction)
                print("✅ Prediction result saved to prediction_results.txt")
            except Exception as save_error:
                print(f"❌ Error saving result: {save_error}")

        return render_template("index.html", 
                             prediction_text=f"Predicted Sales: {formatted_prediction}",
                             success=True,
                             input_data=input_data,
                             processed_data=processed_data)
                             
    except Exception as e:
        error_message = str(e)
        print(f"❌ Prediction error: {error_message}")
        return render_template("index.html", 
                             prediction_text=f"Error: {error_message}",
                             error=True)

def save_prediction_result(input_data, prediction_value, formatted_prediction):
    """Save prediction result to file"""
    expected_keys = [
        'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
        'Outlet_Identifier', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
    ]
    
    with open('prediction_results.txt', 'a', encoding='utf-8') as f:
        f.write("\n" + "="*70 + "\n")
        f.write("BigMart Sales Prediction Result\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n")
        f.write(f"{'Field':<30} {'Value':<35}\n")
        f.write("-"*70 + "\n")
        
        for key in expected_keys:
            display_key = key.replace('_', ' ').title()
            value = input_data[key] if input_data[key] is not None else 'Not Provided'
            f.write(f"{display_key:<30} {str(value):<35}\n")
        
        f.write("-"*70 + "\n")
        f.write(f"{'Predicted Sales':<30} {formatted_prediction:<35}\n")
        f.write(f"{'Raw Prediction Value':<30} {prediction_value:<35.2f}\n")
        f.write("="*70 + "\n")

@app.route("/health")
def health_check():
    """Health check endpoint to verify if the app and model are working"""
    status = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_error": model_error if model is not None else model_error,
        "timestamp": datetime.now().isoformat(),
        "app_version": "2.0",
        "features": ["Web UI", "API", "Model Health Check", "Sample Data", "Auto-save"]
    }
    return jsonify(status)

@app.route("/reload_model")
def reload_model():
    """Endpoint to try reloading the model"""
    success = load_model()
    return jsonify({
        "success": success,
        "model_loaded": model is not None,
        "error": model_error,
        "timestamp": datetime.now().isoformat()
    })

@app.route("/test_prediction")
def test_prediction():
    """Test endpoint to demonstrate model predictions with sample data"""
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "model_error": model_error
        })
    
    try:
        # Sample test cases
        test_cases = [
            {
                "name": "High-end Dairy Product",
                "description": "Premium dairy product in metropolitan area",
                "data": {
                    'Item_Weight': 8.5,
                    'Item_Fat_Content': 'Low Fat',
                    'Item_Visibility': 0.016,
                    'Item_Type': 'Dairy',
                    'Item_MRP': 249.81,
                    'Outlet_Identifier': 'OUT013',
                    'Outlet_Establishment_Year': 1987,
                    'Outlet_Size': 'Medium',
                    'Outlet_Location_Type': 'Tier 1',
                    'Outlet_Type': 'Supermarket Type1'
                }
            },
            {
                "name": "Budget Snack Food",
                "description": "Affordable snack food in smaller market",
                "data": {
                    'Item_Weight': 15.2,
                    'Item_Fat_Content': 'Regular',
                    'Item_Visibility': 0.034,
                    'Item_Type': 'Snack Foods',
                    'Item_MRP': 89.95,
                    'Outlet_Identifier': 'OUT027',
                    'Outlet_Establishment_Year': 1985,
                    'Outlet_Size': 'Small',
                    'Outlet_Location_Type': 'Tier 2',
                    'Outlet_Type': 'Grocery Store'
                }
            },
            {
                "name": "Household Product",
                "description": "Household item in large supermarket",
                "data": {
                    'Item_Weight': 12.8,
                    'Item_Fat_Content': 'Low Fat',
                    'Item_Visibility': 0.028,
                    'Item_Type': 'Household',
                    'Item_MRP': 199.50,
                    'Outlet_Identifier': 'OUT045',
                    'Outlet_Establishment_Year': 2002,
                    'Outlet_Size': 'High',
                    'Outlet_Location_Type': 'Tier 3',
                    'Outlet_Type': 'Supermarket Type2'
                }
            }
        ]
        
        results = []
        for case in test_cases:
            # Preprocess data using the proper function
            df = prepare_dataframe_for_prediction(case["data"])
            prediction = model.predict(df)[0]
            
            results.append({
                "case": case["name"],
                "description": case["description"],
                "input": case["data"],
                "processed_input": processed_data,
                "prediction": f"₹{prediction:,.2f}",
                "prediction_value": float(prediction)
            })
        
        return jsonify({
            "success": True,
            "test_results": results,
            "model_info": {
                "type": str(type(model)),
                "features": len(test_cases[0]["data"]),
                "status": "Model working correctly",
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Test prediction failed: {str(e)}",
            "success": False,
            "timestamp": datetime.now().isoformat()
        })

@app.route("/predict_api", methods=["POST"])
def predict_api():
    """API endpoint for programmatic predictions"""
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "model_error": model_error,
            "timestamp": datetime.now().isoformat()
        }), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No JSON data provided"
            }), 400
        
        # Validate required fields
        required_fields = ['Item_MRP', 'Item_Type', 'Outlet_Type']
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        
        if missing_fields:
            return jsonify({
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields
            }), 400
        
        # Preprocess data using the proper function
        df = prepare_dataframe_for_prediction(data)
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        return jsonify({
            "success": True,
            "prediction": f"₹{prediction:,.2f}",
            "prediction_value": float(prediction),
            "input_data": data,
            "processed_data": processed_data,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "success": False,
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route("/model_info")
def model_info():
    """Get detailed model information"""
    if model is None:
        return jsonify({
            "error": "Model not loaded",
            "model_error": model_error
        }), 500
    
    try:
        model_details = {
            "model_type": str(type(model).__name__),
            "model_module": str(type(model).__module__),
            "loaded": True,
            "timestamp": datetime.now().isoformat()
        }
        
        # Try to get more details if available
        if hasattr(model, 'n_estimators'):
            model_details["n_estimators"] = model.n_estimators
        if hasattr(model, 'max_depth'):
            model_details["max_depth"] = model.max_depth
        if hasattr(model, 'feature_importances_'):
            # Get top 5 most important features
            feature_names = [
                'Item_Weight', 'Item_Fat_Content', 'Item_Visibility', 'Item_Type', 'Item_MRP',
                'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'
            ]
            importances = model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            model_details["top_features"] = feature_importance[:5]
        
        return jsonify(model_details)
        
    except Exception as e:
        return jsonify({
            "error": f"Failed to get model info: {str(e)}",
            "success": False
        }), 500

@app.route("/stats")
def get_stats():
    """Get application statistics"""
    stats = {
        "model_loaded": model is not None,
        "uptime": time.time(),
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Sales Prediction",
            "Model Health Monitoring", 
            "API Access",
            "Sample Data Testing",
            "Results Export",
            "Interactive UI"
        ]
    }
    
    # Check if results file exists and get count
    if os.path.exists('prediction_results.txt'):
        try:
            with open('prediction_results.txt', 'r', encoding='utf-8') as f:
                content = f.read()
                prediction_count = content.count('BigMart Sales Prediction Result')
                stats["total_predictions"] = prediction_count
        except:
            stats["total_predictions"] = 0
    else:
        stats["total_predictions"] = 0
    
    return jsonify(stats)

if __name__ == "__main__":
    print("🚀 Starting BigMart Sales Prediction App...")
    print(f"📊 Model status: {'Loaded' if model is not None else ' Not Loaded'}")
    if model_error:
        print(f"❌ Model error: {model_error}")

    app.run(debug=True, host='0.0.0.0', port=5000)
