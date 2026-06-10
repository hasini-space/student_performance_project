import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_and_prepare_data():
    """
    Simulates loading data from the 'data/' directory.
    Replace this simulation logic with pd.read_csv('data/your_dataset.csv')
    """
    print("[1/5] Loading and simulating student data...")
    np.random.seed(42)
    num_students = 1200

    data = {
        'school_support': np.random.choice(['yes', 'no'], size=num_students, p=[0.25, 0.75]),
        'parent_engagement': np.random.choice(['High', 'Medium', 'Low'], size=num_students, p=[0.4, 0.4, 0.2]),
        'study_time_weekly': np.random.randint(2, 25, size=num_students),
        'absences': np.random.negative_binomial(n=2, p=0.2, size=num_students),
        'midterm_score': np.random.randint(30, 100, size=num_students)
    }
    df = pd.DataFrame(data)
    
    # Calculate target (1 = Pass, 0 = Fail/At-Risk)
    fail_prob = (df['absences'] * 0.04) - (df['study_time_weekly'] * 0.02) - (df['midterm_score'] * 0.015) + 0.4
    df['final_outcome'] = np.where(fail_prob > 0.15, 0, 1)
    
    X = df.drop(columns=['final_outcome'])
    y = df['final_outcome']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def build_preprocessing_pipeline():
    """
    Creates the ColumnTransformer pipeline for text encoding and feature scaling.
    """
    numeric_features = ['study_time_weekly', 'absences', 'midterm_score']
    categorical_features = ['school_support', 'parent_engagement']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ]
    )
    return preprocessor, numeric_features, categorical_features

def main():
    # 1. Load Data
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    # 2. Preprocess Data
    print("[2/5] Initializing and fitting preprocessing pipeline...")
    preprocessor, num_cols, cat_cols = build_preprocessing_pipeline()
    
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Extract clean feature names for post-analysis
    encoded_cat_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
    feature_names = num_cols + list(encoded_cat_cols)

    # 3. Model Training & Hyperparameter Tuning
    print("[3/5] Tuning Random Forest via Grid Search...")
    rf_base = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8],
        'min_samples_split': [2, 5]
    }
    
    # Optimize for recall/f1 to catch at-risk students reliably
    grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train_transformed, y_train)
    
    best_model = grid_search.best_estimator_
    print(f"Best Hyperparameters Found: {grid_search.best_params_}")

    # 4. Evaluation
    print("\n[4/5] Evaluating model on unseen test split...")
    y_pred = best_model.predict(X_test_transformed)
    y_prob = best_model.predict_proba(X_test_transformed)[:, 1]

    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=['At-Risk (Fail)', 'Passing']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))

    print("\n=== TOP FEATURE IMPORTANCES ===")
    importances = best_model.feature_importances_
    for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"{name:<30} : {importance:.4f}")

    # 5. Save Artifacts
    print("\n[5/5] Saving model and preprocessor serialization artifacts...")
    os.makedirs('models', exist_ok=True)
    
    # Save the pipeline and model separately to keep production deployment modular
    joblib.dump(best_model, 'models/random_forest_model.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Success! Artifacts saved safely to 'models/' directory.")

if __name__ == '__main__':
    main()