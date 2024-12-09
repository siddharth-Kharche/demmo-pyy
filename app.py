import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from ml_model import preprocess_input, generate_improvement_suggestions

app = Flask(__name__)

# Load model and preprocessors
def load_model_and_preprocessors():
    try:
        model = joblib.load('random_forest_school_status_model (5).joblib')
        le = joblib.load('label_encoder (7).joblib')
        scaler = joblib.load('feature_scaler (7).joblib')
        imputer = joblib.load('feature_imputer (5).joblib')
        return model, le, scaler, imputer
    except FileNotFoundError as e:
        print(f"FileNotFoundError: {e}")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading model and preprocessors: {e}")
        return None, None, None, None

# Global model loading
MODEL, LABEL_ENCODER, SCALER, IMPUTER = load_model_and_preprocessors()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Collect input data from form
        input_data = {
            'Total Class Rooms': int(request.form['total_class_rooms']),
            'Total Teachers': int(request.form['total_teachers']),
            'Total Students': int(request.form['total_students']),
            'Boundary Wall': int(request.form.get('boundary_wall', 0)),
            'Library Available': int(request.form.get('library_available', 0)),
            'Separate Room for HM': int(request.form.get('separate_hm_room', 0)),
            'Drinking Water Available': int(request.form.get('drinking_water', 0)),
            'Playground Available': int(request.form.get('playground', 0)),
            'Electricity Availability': int(request.form.get('electricity', 0)),
            'Grade Configuration': request.form.get('grade_configuration', '1,12'),
            'Total Washrooms': request.form.get('total_washrooms', '2')
        }

        if not all([MODEL, LABEL_ENCODER, SCALER, IMPUTER]):
            return render_template('index.html', error="Model or preprocessors failed to load. Check logs for details.")

        try:
            # Preprocess and predict
            X_processed = preprocess_input(input_data, IMPUTER, SCALER)
            prediction = MODEL.predict(X_processed)
            pred_proba = MODEL.predict_proba(X_processed)
            pred_label = LABEL_ENCODER.inverse_transform(prediction)[0]

            # Calculate probabilities
            prob_df = pd.DataFrame({
                'Status': LABEL_ENCODER.classes_,
                'Probability': pred_proba[0]
            }).sort_values('Probability', ascending=False)

            # Generate improvement suggestions if not standard
            suggestions = ""
            if pred_label.lower() != 'standard':
                suggestions = generate_improvement_suggestions(input_data, pred_label)

            return render_template('result.html', 
                                   prediction=pred_label, 
                                   probabilities=prob_df, 
                                   input_data=input_data,
                                   suggestions=suggestions)

        except Exception as e:
            return render_template('index.html', error=f"An error occurred during processing: {str(e)}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

