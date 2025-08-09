# ❤️ Heart Disease Prediction Project

This project applies machine learning techniques to the UCI Heart Disease dataset to predict the presence of heart disease.

---

## 📁 Project Structure

```text
Heart_Disease_Project/
│
├── data/                    # Raw and processed data
│   └── heart_disease.csv
│
├── notebooks/               # Jupyter Notebooks for each step
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pca_analysis.ipynb
│   ├── 03_feature_selection.ipynb
│   ├── 04_supervised_learning.ipynb
│   ├── 05_unsupervised_learning.ipynb
│   └── 06_hyperparameter_tuning.ipynb
│
├── models/                  # Saved models
│   ├── final_model.pkl
│   └── scaler.pkl
│
├── ui/                      # Streamlit app (optional)
│   └── app.py
│
├── deployment/              # Ngrok or deployment instructions
│   └── ngrok_setup.txt
│
├── results/                 # Evaluation metrics
│   └── evaluation_metrics.txt
│
├── requirements.txt         # Required packages
└── README.md                # Project overview
```
*No deployment applied. Streamlit app was not included in this project.*

---

## 📝 Project Overview

This repository contains a full machine learning pipeline for heart disease prediction, including:

- **Data preprocessing**: Cleaning, encoding, and scaling.
- **Dimensionality reduction**: PCA analysis and visualization.
- **Feature selection**: Using statistical and model-based techniques.
- **Supervised learning**: Training and evaluating various classifiers (Logistic Regression, Decision Tree, Random Forest, SVM).
- **Unsupervised learning**: Clustering (KMeans, Hierarchical) and cluster evaluation.
- **Hyperparameter tuning**: Grid search and random search for model optimization.

---

## 🚀 How to Run

1. **Clone the repository**  
   ```bash
   git clone <repo-url>
   cd Heart_Disease_Project
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks in order:**  
   - Start from `01_data_preprocessing.ipynb` and continue sequentially.
   - Each notebook builds on the output of the previous one.

---

## 📚 Notebooks Description

- **01_data_preprocessing.ipynb**  
  Load the dataset, handle missing values, encode categorical variables, and scale features.

- **02_pca_analysis.ipynb**  
  Apply PCA, visualize explained variance and 2D projections.

- **03_feature_selection.ipynb**  
  Select important features using Chi2, RFE, and Random Forest importance.

- **04_supervised_learning.ipynb**  
  Train and evaluate classifiers; compare metrics and visualize ROC curves.

- **05_unsupervised_learning.ipynb**  
  Perform clustering, use the elbow method, and visualize dendrograms.

- **06_hyperparameter_tuning.ipynb**  
  Tune model hyperparameters and save the best model.

---

## ✅ Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- ucimlrepo
- joblib

*(see `requirements.txt` for full list)*

---

## 📊 Results & Visualizations

- **PCA plots**: Show variance explained, 2D separation.
- **Feature importance**: Top features for prediction.
- **Model metrics**: Accuracy, Precision, Recall, F1, ROC AUC.
- **ROC curves**: Compare classifier performance.
- **Clustering**: Elbow method, ARI score, dendrograms.

---

## 👨‍💻 Notes from the Author

During this project, I faced some challenges with feature selection, especially with the Chi2 test requiring all input values to be non-negative. I solved this using MinMaxScaler.  
I also experimented a lot with PCA and clustering—sometimes the clusters made sense, sometimes not!  
If you have ideas for improvement or spot mistakes, feel free to open an issue or contact me.

---


## 🙏 Credits

- Dataset: [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- Author: Omar Atef (@Omaratef-gh)

---

## 📝 Notes

- Folders (`data/`, `models/`, etc.) are auto-created by the code as needed.
- Always run notebooks in order for correct results.
- No deployment or Streamlit app is included by default.



## 🚫 Deployment & UI

No deployment or Streamlit web app was included in this project.  
All results are available through the Jupyter Notebooks and output files.