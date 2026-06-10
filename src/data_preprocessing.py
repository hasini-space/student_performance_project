import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_mock_raw_data(output_path="data/raw_students.csv"):
    """
    Generates a messy, real-world style raw dataset containing missing values 
    and anomalies to demonstrate proper data cleaning pipelines.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(42)
    num_students = 1200

    data = {
        'school_support': np.random.choice(['yes', 'no', None], size=num_students, p=[0.23, 0.72, 0.05]),
        'parent_engagement': np.random.choice(['High', 'Medium', 'Low', np.nan], size=num_students, p=[0.38, 0.38, 0.19, 0.05]),
        'study_time_weekly': np.random.randint(2, 25, size=num_students),
        'absences': np.random.negative_binomial(n=2, p=0.2, size=num_students),
        'midterm_score': np.random.randint(30, 100, size=num_students)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce explicit anomalies (e.g., negative absences or impossible test scores)
    df.loc[10, 'absences'] = -5 
    df.loc[25, 'midterm_score'] = 150 
    
    # Compute deterministic target column
    fail_prob = (df['absences'] * 0.04) - (df['study_time_weekly'] * 0.02) - (df['midterm_score'] * 0.015) + 0.4
    df['final_outcome'] = np.where(fail_prob > 0.15, 0, 1)
    
    df.to_csv(output_path, index=False)
    print(f"📦 Simulated raw messy data written to: {output_path}")

def clean_and_preprocess_dataset(input_path="data/raw_students.csv"):
    """
    Loads raw tabular data, cleans anomalies, handles missing data points,
    and returns stratified data matrices.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Missing raw target data file at {input_path}")
        
    df = pd.read_csv(input_path)
    print(f"⚙️ Raw data loaded. Initial shape: {df.shape}")
    
    # 1. Handle Structural Anomalies / Outliers
    # Clip negative absences to zero, clamp scores over 100 back to maximum limits
    df['absences'] = df['absences'].clip(lower=0)
    df['midterm_score'] = df['midterm_score'].clip(upper=100)
    
    # 2. Impute Categorical Missing Values via Mode (Most frequent entry)
    for col in ['school_support', 'parent_engagement']:
        if df[col].isnull().sum() > 0:
            most_frequent = df[col].mode()[0]
            df[col] = df[col].fillna(most_frequent)
            print(f"   Filled missing elements in '{col}' using mode: '{most_frequent}'")

    # 3. Stratified Train-Test Splitting
    X = df.drop(columns=['final_outcome'])
    y = df['final_outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed components back to disk for reference or custom training runs
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    
    print("✅ Cleaning complete. Structured data subsets exported to 'data/' folder.")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # If running independently, generate mock file and run cleaning lifecycle
    create_mock_raw_data()
    clean_and_preprocess_dataset()