import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import pickle
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', '  ', resume_text)
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    resume_text = re.sub(r'\d+', '', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text).strip()
    return resume_text

print("Loading and preprocessing data...")
df = pd.read_csv('UpdatedResumeDataSet.csv')
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

# Identify categories with too few samples
category_counts = df['Category'].value_counts()
min_samples_per_category = 2
valid_categories = category_counts[category_counts >= min_samples_per_category].index

print(f"Categories with at least {min_samples_per_category} samples: {len(valid_categories)}")
print("Categories removed due to insufficient samples:")
print(category_counts[category_counts < min_samples_per_category])

# Filter the dataframe to keep only valid categories
df_filtered = df[df['Category'].isin(valid_categories)]

custom_stop_words = ['experience', 'skill', 'work', 'project', 'developed', 'implemented']
stop_words = list(set(stopwords.words('english')).union(custom_stop_words))

print("Extracting features...")
tfidf = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 2),
    max_features=10000,
    max_df=0.5
)
features = tfidf.fit_transform(df_filtered['cleaned_resume'])

le = LabelEncoder()
labels = le.fit_transform(df_filtered['Category'])

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

print("Applying SMOTE for oversampling...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Training the model...")
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
clf.fit(X_train_resampled, y_train_resampled)

print("Evaluating the model...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Saving the model and vectorizer...")
pickle.dump(clf, open('clf_v3.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_v3.pkl', 'wb'))
pickle.dump(le, open('label_encoder_v3.pkl', 'wb'))

print("Training complete. New model and vectorizer saved as clf_v3.pkl and tfidf_v3.pkl")