import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. LOAD & SAMPLE (Using 10k rows for efficiency)
df = pd.read_csv('global_ai_jobs.csv').sample(10000, random_state=42)

# 2. PREPROCESSING
# Encode categorical text into numbers
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if col != 'experience_level':
        df[col] = le.fit_transform(df[col])

# Prepare Target (X=Features, y=Target)
X = df.drop(['id', 'experience_level'], axis=1)
y = le.fit_transform(df['experience_level'])

# SCALING (Mandatory for SVM and Neural Networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. MODELS
models = {
    "SVM (Linear SVC)": LinearSVC(max_iter=10000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.01) # alpha is the regularization
}

# 4. TRAINING & EVALUATION
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    print(f"{name} trained successfully.")

# 5. VISUALIZATION: Compare Results
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.values()), y=list(results.keys()), palette='magma')
plt.title('AI Jobs: Model Accuracy Comparison')
plt.show()

# 6. VISUALIZATION: SVM Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, models["SVM (Linear SVC)"].predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('SVM Prediction Confusion Matrix')
plt.show()

# Final Statistics
print("\n--- PERFORMANCE STATISTICS (SVM) ---")
print(classification_report(y_test, models["SVM (Linear SVC)"].predict(X_test), target_names=le.classes_))