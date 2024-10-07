import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re

def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', '  ', resume_text)
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    resume_text = re.sub(r'\d+', '', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text).strip()
    return resume_text

# Load the model, vectorizer, and label encoder
clf = pickle.load(open('clf_v3.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_v3.pkl', 'rb'))
le = pickle.load(open('label_encoder_v3.pkl', 'rb'))

# Load and preprocess the data
print("Loading and preprocessing data...")
df = pd.read_csv('UpdatedResumeDataSet.csv')
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Filter categories (if needed, based on your training script)
category_counts = df['Category'].value_counts()
min_samples_per_category = 2
valid_categories = category_counts[category_counts >= min_samples_per_category].index
df_filtered = df[df['Category'].isin(valid_categories)]

# Prepare X and y
X = tfidf.transform(df_filtered['cleaned_resume'])
y = le.transform(df_filtered['Category'])

# Perform cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(clf, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

# Feature importance (for RandomForest)
print("\nAnalyzing feature importance...")
feature_importance = clf.estimators_[0].feature_importances_
feature_names = tfidf.get_feature_names_out()
feature_importance_dict = dict(zip(feature_names, feature_importance))
top_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]

print("Top 20 important features:")
for feature, importance in top_features:
    print(f"{feature}: {importance:.4f}")

# Confusion Matrix
print("\nGenerating confusion matrix...")
y_pred = clf.predict(X)
cm = confusion_matrix(y, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()

print("Confusion matrix saved as 'confusion_matrix.png'")

# Additional analysis: Print classification report
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=le.classes_))