import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file, url_for, redirect
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Define base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and scaler
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'random_forest_model.joblib')
model = joblib.load(MODEL_PATH)
scaler = joblib.load(os.path.join(BASE_DIR, 'model', 'scaler.joblib'))

# Load dataset for preview and dropdown options
df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'heart.csv'))

# Get feature names and ranges
FEATURES = df.columns[:-1].tolist()
FEATURE_RANGES = {}
FEATURE_OPTIONS = {}

# Get ranges for numerical features and unique values for categorical features
for feature in FEATURES:
    if feature in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
        FEATURE_RANGES[feature] = [df[feature].min(), df[feature].max()]
    else:
        FEATURE_OPTIONS[feature] = df[feature].unique()

# FIXED: Convert dataset preview to list of lists (not list of dicts)
DATASET_COLUMNS = df.columns.tolist()
DATASET_ROWS = df.head(20).values.tolist()  # FIXED LINE

# Define prediction labels
LABELS = {
    0: 'No Disease',
    1: 'Has Disease'
}

@app.route('/visualizations/<filename>')
def show_visualization(filename):
    img_path = os.path.join(BASE_DIR, 'visualizations', filename)
    return send_file(img_path, mimetype='image/png')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            input_data = []
            for feature in FEATURES:
                value = request.form.get(feature)
                if value is None:
                    raise ValueError(f"Missing value for {feature}")
                input_data.append(float(value))
        except ValueError as e:
            error_msg = str(e)
            if "could not convert string to float" in error_msg:
                error_msg = "Please enter valid numbers for all fields"
            return render_template('index.html', 
                                   error=error_msg,
                                   features=FEATURES,
                                   feature_ranges=FEATURE_RANGES,
                                   feature_options=FEATURE_OPTIONS,
                                   dataset_columns=DATASET_COLUMNS,
                                   dataset_rows=DATASET_ROWS)

        # Preprocess
        X = scaler.transform([input_data])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # Create feature-value pairs
        feature_values = dict(zip(FEATURES, input_data))

        return redirect(url_for('result', prediction=prediction, probability=proba[1], **feature_values))

    return render_template('index.html',
                           features=FEATURES,
                           feature_ranges=FEATURE_RANGES,
                           feature_options=FEATURE_OPTIONS,
                           dataset_columns=DATASET_COLUMNS,
                           dataset_rows=DATASET_ROWS)

@app.route('/result')
def result():
    prediction = request.args.get('prediction')
    probability = float(request.args.get('probability'))

    # Get feature-value pairs from query string
    feature_values = {}
    for feature in FEATURES:
        try:
            value = float(request.args.get(feature))
            feature_values[feature] = value
        except (ValueError, TypeError):
            feature_values[feature] = None

    return render_template(
        'result.html',
        label='Has Heart Disease' if prediction == '1' else 'No Heart Disease',
        proba=[1 - probability, probability],
        features=FEATURES,
        feature_values=feature_values
    )

@app.route('/visualizations/<img>')
def show_img(img):
    img_path = os.path.join(BASE_DIR, 'visualizations', img)
    return send_file(img_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
