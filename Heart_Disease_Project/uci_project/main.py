"""
Heart Disease Prediction & Clustering Pipeline
Author: Omaratef-gh
"""

# ---- CONFIGURATION ----
UCI_DATASET_ID = 45
N_TOP_FEATURES = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---- IMPORTS ----
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, roc_curve, auc, adjusted_rand_score
)
from scipy.stats import randint
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib

# ---- DATA LOADING ----
heart_disease = fetch_ucirepo(id=UCI_DATASET_ID)
X_raw = heart_disease.data.features
y_raw = heart_disease.data.targets

print(heart_disease.metadata)
print(heart_disease.variables)

df = pd.concat([X_raw, y_raw], axis=1)
print(df.head(), df.shape)
print(df.dtypes)
print(df.isnull().sum())

# ---- PREPROCESSING ----
df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop('num', axis=1).columns.tolist()
print("Categorical:", categorical_cols)
print("Numerical:", numerical_cols)

# Impute missing values for numerical columns
imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

# One-hot encode categorical columns
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Scale numerical columns
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

X = df_encoded.drop('num', axis=1)
y = df_encoded['num']

# ---- EXPLORATORY DATA ANALYSIS ----
df[numerical_cols].hist(bins=20, figsize=(15, 10))
plt.suptitle('Distribution of Numerical Features')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='num', y=col, data=df)
    plt.title(f'{col} vs Heart Disease')
    plt.show()

# ---- FEATURE SELECTION ----
scaler_fs = MinMaxScaler()
X_scaled = scaler_fs.fit_transform(X)

chi_selector = SelectKBest(score_func=chi2, k=N_TOP_FEATURES)
X_selected = chi_selector.fit_transform(X_scaled, y)
selected_columns = X.columns[chi_selector.get_support()]
print("Top Features by Chi-Square Test:", list(selected_columns))
X_selected_df = pd.DataFrame(X_selected, columns=selected_columns)

# ---- DIMENSIONALITY REDUCTION ----
pca = PCA()
X_pca = pca.fit_transform(X_selected_df)

plt.figure(figsize=(10,6))
plt.plot(range(1, len(pca.explained_variance_ratio_)+1),
         pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_selected_df)
df_pca = pd.DataFrame(X_pca_2, columns=['PC1', 'PC2'])
df_pca['target'] = y.values

plt.figure(figsize=(10,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='target', palette='viridis')
plt.title('PCA - 2D Projection of Heart Disease Dataset')
plt.show()

# Use 2D PCA for downstream to keep clustering/visualization simple
X_final = pd.DataFrame(X_pca_2, columns=['PC1', 'PC2'])
y_final = y

# ---- RANDOM FOREST FEATURE IMPORTANCE (on selected features) ----
rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf.fit(X_final, y_final)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10,6))
plt.title("Feature Importances (Random Forest)")
plt.bar(range(X_final.shape[1]), importances[indices])
plt.xticks(range(X_final.shape[1]), X_final.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# ---- MODEL TRAINING & EVALUATION ----
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=TEST_SIZE, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)
}
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
    results[name] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC AUC": roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"
    }
    print(f"----- {name} -----")
    print(classification_report(y_test, y_pred))
    print("\n")

results_df = pd.DataFrame(results).T
print("🔍 Model Comparison:")
print(results_df)

# ---- ROC CURVE ----
plt.figure(figsize=(10,6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid()
plt.show()

# ---- CLUSTERING ----
from sklearn.cluster import KMeans

inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE)
    kmeans.fit(X_final)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.grid()
plt.show()

kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE)
clusters = kmeans.fit_predict(X_final)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_final.iloc[:,0], y=X_final.iloc[:,1], hue=clusters, palette='viridis')
plt.title('K-Means Clustering Result')
plt.show()

# Hierarchical clustering
linked = linkage(X_final, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode='lastp', p=30, leaf_rotation=90.)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()

print(f"Adjusted Rand Index (KMeans vs True Labels): {adjusted_rand_score(y_final, clusters):.4f}")

# ---- HYPERPARAMETER TUNING ----
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters (GridSearchCV):", grid_search.best_params_)
print("Best accuracy on validation set:", grid_search.best_score_)

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 5)
}
random_search = RandomizedSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_distributions=param_dist,
                                   n_iter=20, cv=5, scoring='accuracy', random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
random_search.fit(X_train, y_train)
print("Best parameters (RandomizedSearchCV):", random_search.best_params_)
print("Best accuracy on validation set:", random_search.best_score_)

# ---- FINAL MODEL SAVE ----
best_model = random_search.best_estimator_
best_model.fit(X_train, y_train)
joblib.dump(best_model, 'final_heart_model.pkl')
print("✅ Model saved as final_heart_model.pkl")
joblib.dump(scaler, 'scaler.pkl')
print("✅ Scaler saved as scaler.pkl")