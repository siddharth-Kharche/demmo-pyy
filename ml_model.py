import pandas as pd
import numpy as np
from groq import Groq
import os
def safe_numeric_convert(value):
    """Safely convert input to numeric, handling empty or non-numeric values."""
    try:
        return float(str(value).strip() if value else 0)
    except ValueError:
        return 0

def preprocess_input(input_data, imputer, scaler):
    """Preprocess input data for model prediction."""
    if imputer is None or scaler is None:
        raise ValueError("Preprocessors are not loaded. Ensure imputer and scaler are initialized.")

    # Convert input data into a DataFrame
    df = pd.DataFrame([input_data])

    # Handle 'Grade Configuration'
    grade_config = input_data['Grade Configuration'].split(',')
    df['Grade_Start'] = safe_numeric_convert(grade_config[0])
    df['Grade_End'] = safe_numeric_convert(grade_config[1]) if len(grade_config) > 1 else 0

    df['Total_Washrooms'] = safe_numeric_convert(input_data['Total Washrooms'])

    # Feature selection and preprocessing
    features = [
        'Boundary Wall', 'Total Class Rooms', 'Library Available',
        'Separate Room for HM', 'Drinking Water Available',
        'Playground Available', 'Electricity Availability',
        'Total Teachers', 'Total_Washrooms', 'Total Students',
        'Grade_Start', 'Grade_End'
    ]

    df = df[features]
    X_imputed = pd.DataFrame(imputer.transform(df), columns=features)
    X_scaled = scaler.transform(X_imputed)

    return X_scaled

def generate_improvement_suggestions(input_data, prediction_label):
    """Generate school improvement suggestions using Groq Cloud LLM."""
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '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')  # Replace securely

    try:
        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
        You are an expert educational consultant analyzing a school classified as {prediction_label}.
        School Details: {...}
        Provide actionable improvement suggestions to meet standard criteria.
        """
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an educational consultant."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192"
        )

        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"Error generating suggestions: {str(e)}"
