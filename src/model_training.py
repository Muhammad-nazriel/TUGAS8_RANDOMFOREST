import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def summarize_dataset(df, project_root):
    """
    Generate summary statistics of the heart disease dataset
    """
    print("\n=== Heart Disease Dataset Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Number of features: {len(df.columns) - 1}")  # Exclude target
    print("\nFeature statistics:")
    print(df.describe().round(2))
    
    print("\nTarget distribution:")
    target_dist = df['target'].value_counts()
    print(target_dist)
    
    # Save target distribution plot
    plt.figure(figsize=(10, 6))
    ax = target_dist.plot(kind='bar', color=['red', 'green'])
    plt.title('Heart Disease Distribution', pad=20, fontsize=14)
    plt.xlabel('Target', labelpad=20, fontsize=12)
    plt.ylabel('Number of Patients', labelpad=20, fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['No Disease', 'Has Disease'], fontsize=10)
    plt.yticks(fontsize=10)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(str(p.get_height()), 
                   (p.get_x() * 1.005, p.get_height() * 1.005),
                   fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'visualizations', 'heart_disease_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save correlation matrix
    plt.figure(figsize=(12, 8))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix', pad=20, fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'visualizations', 'correlation_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def load_and_preprocess_data(project_root):
    # Load dataset
    df = pd.read_csv(os.path.join(project_root, 'data', 'heart.csv'))
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, os.path.join(project_root, 'model', 'scaler.joblib'))
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_model(X_train, y_train, project_root):
    # Initialize and train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Save model
    joblib.dump(rf, os.path.join(project_root, 'model', 'random_forest_model.joblib'))
    return rf

def evaluate_model(model, X_train, X_test, y_test, project_root):
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Disease', 'Has Disease']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['No Disease', 'Has Disease'],
               yticklabels=['No Disease', 'Has Disease'])
    plt.title('Confusion Matrix', pad=20, fontsize=14)
    plt.xlabel('Predicted', labelpad=20, fontsize=12)
    plt.ylabel('True', labelpad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'visualizations', 'confusion_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature Importance
    # Get feature names from the original dataset
    df = pd.read_csv(os.path.join(project_root, 'data', 'heart.csv'))
    feature_names = df.drop('target', axis=1).columns
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    # Create horizontal bar plot with custom styling
    ax = plt.barh(feature_importance['feature'], 
                 feature_importance['importance'],
                 color='skyblue')
    
    plt.title('Feature Importance', pad=20, fontsize=14)
    plt.xlabel('Importance Score', labelpad=20, fontsize=12)
    plt.ylabel('Features', labelpad=20, fontsize=12)
    
    # Add value labels
    for i, v in enumerate(feature_importance['importance']):
        plt.text(v + 0.01, i, f'{v:.4f}', color='black', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'visualizations', 'feature_importance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return accuracy

def main():
    # Get absolute path to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create necessary directories
    os.makedirs(os.path.join(project_root, 'model'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'visualizations'), exist_ok=True)
    
    # Load and summarize dataset
    df = pd.read_csv(os.path.join(project_root, 'data', 'heart.csv'))
    summarize_dataset(df, project_root)
    
    # Data preprocessing
    X_train, X_test, y_train, y_test = load_and_preprocess_data(project_root)
    
    # Model training
    model = train_model(X_train, y_train, project_root)
    
    # Model evaluation
    accuracy = evaluate_model(model, X_train, X_test, y_test, project_root)
    print(f"\nFinal Model Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
