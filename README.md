# Leaf Classification System using Random Forest

This project implements a Random Forest classifier to classify leaf species based on morphological features. The system includes both model training and a web application interface.

## Project Structure

```
LeafClassification_RF/
├── data/
│   └── train.csv
├── src/
│   └── model_training.py
├── app/
│   ├── static/
│   │   └── style.css
│   ├── templates/
│   │   ├── index.html
│   └── app.py
├── model/
│   ├── random_forest_model.joblib
│   ├── scaler.joblib
│   └── label_encoder.joblib
├── visualizations/
│   ├── confusion_matrix.png
│   └── feature_importance.png
├── requirements.txt
└── README.md
```

## Features

- Random Forest classification of leaf species
- Feature importance visualization
- Confusion matrix visualization
- Web application interface using Flask and Bootstrap
- Real-time classification results

## Setup Instructions

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Place the train.csv dataset in the `data` directory

3. Train the model:
```bash
python src/model_training.py
```

4. Run the web application:
```bash
cd app
python app.py
```

5. Access the application at http://localhost:5000

## Dataset

The dataset contains 990 samples with 192 features each:
- margin1-margin64: edge shape features
- shape1-shape64: overall shape features
- texture1-texture64: texture features
- species: target variable (99 different species)

## Model Performance

The model uses Random Forest with 100 estimators and provides:
- Accuracy metrics
- Classification report
- Confusion matrix visualization
- Feature importance visualization

## Web Application

The web application provides a simple interface to:
1. Input leaf features
2. Get real-time classification results
3. View predicted species

## License

This project is licensed under the MIT License - see the LICENSE file for details.
