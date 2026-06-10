# 🎓 Student Performance Predictive Analytics

A machine learning system that helps educators identify at-risk students early in the academic term using predictive analytics. This project combines data preprocessing, model training, and an interactive web dashboard to support student success initiatives.

## 🚀 Live Demo

**Try it now:** [https://hxvahgvoydyxgbf9sndkeg.streamlit.app/](https://hxvahgvoydyxgbf9sndkeg.streamlit.app/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project leverages machine learning to predict student performance outcomes (pass/fail) based on academic and institutional factors. The system is designed to support early intervention strategies and improve student success rates.

**Key Benefits:**
- 🚀 **Early Identification**: Catch at-risk students before they fail
- 📊 **Data-Driven Insights**: Make informed intervention decisions
- 💡 **Actionable Recommendations**: Get specific guidance for each student
- 🎯 **Institutional Impact**: Support student success rates

---

## ✨ Features

### 1. **Smart Data Preprocessing**
   - Handles missing values intelligently
   - Detects and corrects data anomalies
   - Automatic feature scaling and encoding
   - Stratified train-test splitting for balanced evaluation

### 2. **Advanced Machine Learning**
   - Random Forest classifier with hyperparameter tuning
   - Grid search optimization for F1 score
   - Comprehensive model evaluation metrics
   - Feature importance analysis

### 3. **Interactive Web Dashboard**
   - Built with Streamlit for ease of use
   - Real-time prediction interface
   - Visual probability breakdowns
   - Contextual intervention recommendations
   - Professional, educator-friendly UI

### 4. **Production-Ready**
   - Model serialization with joblib
   - Modular architecture for easy updates
   - Caching for optimal performance
   - Clear error handling

---

## 📁 Project Structure

```
student_performance_project/
├── src/
│   ├── app.py                    # Streamlit web dashboard
│   ├── train_model.py            # Model training and evaluation
│   └── data_preprocessing.py     # Data cleaning and preparation
├── data/
│   ├── raw_students.csv          # Generated raw dataset
│   ├── X_train.csv               # Training features
│   ├── X_test.csv                # Testing features
│   ├── y_train.csv               # Training labels
│   └── y_test.csv                # Testing labels
├── models/
│   ├── random_forest_model.pkl   # Trained model artifact
│   └── preprocessor.pkl          # Data preprocessing pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/hasini-space/student_performance_project.git
   cd student_performance_project
   ```

2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚡ Quick Start

### Step 1: Generate Data & Train Model
```bash
python src/train_model.py
```

This command will:
- Generate a simulated dataset of 1,200 students
- Preprocess the data
- Train and tune a Random Forest model
- Save the model and preprocessor to the `models/` folder
- Display comprehensive evaluation metrics

**Expected Output:**
```
[1/5] Loading and simulating student data...
[2/5] Initializing and fitting preprocessing pipeline...
[3/5] Tuning Random Forest via Grid Search...
Best Hyperparameters Found: {...}
[4/5] Evaluating model on unseen test split...
[5/5] Saving model and preprocessor serialization artifacts...
Success! Artifacts saved safely to 'models/' directory.
```

### Step 2: Launch the Dashboard
```bash
streamlit run src/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### Using the Web Dashboard

1. **Adjust Student Parameters** in the sidebar:
   - **Midterm Exam Score** (0-100)
   - **Weekly Study Hours** (0-40)
   - **Number of Absences** (0-50)
   - **School Support** (Yes/No)
   - **Parental Engagement** (High/Medium/Low)

2. **View Real-Time Predictions**:
   - Prediction status (Passing or At-Risk)
   - Probability scores
   - Visual probability breakdown

3. **Review Intervention Recommendations**:
   - Contextual guidance based on student profile
   - Specific action items for educators
   - Risk factors highlighted

### Using the API Programmatically

```python
import joblib
import pandas as pd

# Load artifacts
model = joblib.load('models/random_forest_model.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')

# Prepare student data
student_data = pd.DataFrame([{
    'school_support': 'yes',
    'parent_engagement': 'High',
    'study_time_weekly': 15,
    'absences': 2,
    'midterm_score': 85
}])

# Get prediction
X_transformed = preprocessor.transform(student_data)
prediction = model.predict(X_transformed)
probabilities = model.predict_proba(X_transformed)

print(f"Prediction: {'Pass' if prediction[0] == 1 else 'At-Risk'}")
print(f"Pass Probability: {probabilities[0][1]:.2%}")
```

---

## 🔧 Technical Details

### Data Features

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| `midterm_score` | Integer | 0-100 | Student's midterm exam score |
| `study_time_weekly` | Integer | 0-40 | Weekly study hours |
| `absences` | Integer | 0+ | Number of class absences |
| `school_support` | Categorical | yes/no | Receiving extra support |
| `parent_engagement` | Categorical | High/Medium/Low | Parental involvement level |

### Target Variable

- **`final_outcome`**: Binary classification
  - `1` = Passing
  - `0` = At-Risk (Fail)

### Model Architecture

- **Algorithm**: Random Forest Classifier
- **Features**: 7 (after one-hot encoding)
- **Hyperparameters**: Tuned via GridSearchCV
- **Preprocessing**: StandardScaler (numeric) + OneHotEncoder (categorical)

### Key Hyperparameters

```python
{
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5]
}
```

---

## 📊 Model Performance

The model is optimized for the **F1 score** to balance precision and recall, ensuring we reliably catch at-risk students while minimizing false alarms.

**Typical Performance Metrics:**
- Classification report with precision, recall, and F1 scores
- ROC-AUC score for overall discriminative ability
- Confusion matrix showing true/false positives and negatives
- Feature importance rankings

To view detailed performance metrics:
```bash
python src/train_model.py
```

The output includes comprehensive evaluation metrics and feature importance analysis.

---

## 🎓 Educational Use Cases

### 1. **Early Intervention Programs**
Identify students who may benefit from tutoring, mentoring, or counseling services.

### 2. **Resource Allocation**
Direct support services to students with the highest need.

### 3. **Academic Planning**
Inform decisions about course recommendations and academic pathways.

### 4. **Institutional Research**
Understand which factors most strongly influence student success.

---

## 📦 Dependencies

```
pandas          # Data manipulation and analysis
numpy           # Numerical computing
scikit-learn    # Machine learning algorithms
joblib          # Model serialization
streamlit       # Interactive web dashboard
matplotlib      # Statistical visualization
seaborn         # Enhanced data visualization
```

For exact versions, see `requirements.txt`

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

### Areas for Contribution:
- Additional machine learning models
- Enhanced data visualization
- More sophisticated preprocessing techniques
- Additional intervention recommendations
- Documentation improvements

---

## 📝 License

This project is open source and available under the MIT License.

---

## 📞 Support & Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Review the code documentation in each module
- Check the Streamlit dashboard for helpful error messages

---

## 🎯 Future Enhancements

- [ ] Integration with real student databases
- [ ] Time-series analysis for semester progression
- [ ] Multi-class outcome prediction (Pass/At-Risk/Excellent)
- [ ] Advanced visualizations with plotly
- [ ] Model explainability with SHAP values
- [ ] Automated model retraining pipeline
- [ ] API endpoint for batch predictions
- [ ] User authentication and multi-tenant support

---

## 📚 References & Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

---

**Last Updated**: June 2026  
**Python Version**: 3.8+  
**Status**: Active Development
