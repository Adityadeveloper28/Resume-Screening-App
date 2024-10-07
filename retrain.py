import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pickle
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)

def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', '  ', resume_text)
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    resume_text = re.sub(r'\d+', '', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text).strip().lower()
    return resume_text

print("Loading and preprocessing data...")
df = pd.read_csv('UpdatedResumeDataSet.csv')
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

custom_stop_words = ['experience', 'skill', 'work', 'project', 'developed', 'implemented']
stop_words = list(set(stopwords.words('english')).union(custom_stop_words))

print("Extracting features...")
tfidf = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 3),  # Include up to trigrams
    max_features=10000,  # Increase number of features
    max_df=0.5
)
features = tfidf.fit_transform(df['cleaned_resume'])

le = LabelEncoder()
labels = le.fit_transform(df['Category'])

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

print("Original class distribution:")
print(Counter(y_train))

print("Balancing the dataset...")
# Only oversample classes with less than 20 samples to a maximum of 20
sampling_strategy = {k: min(v, 20) for k, v in Counter(y_train).items() if v < 20}
over = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train_over, y_train_over = over.fit_resample(X_train, y_train)

print("Class distribution after oversampling:")
print(Counter(y_train_over))

# Undersample majority classes to a maximum of 50 samples
sampling_strategy = {k: min(v, 50) for k, v in Counter(y_train_over).items()}
under = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
X_train_resampled, y_train_resampled = under.fit_resample(X_train_over, y_train_over)

print("Final class distribution after under-sampling:")
print(Counter(y_train_resampled))

print("Training the model...")
clf = RandomForestClassifier(n_estimators=50, max_depth=8, min_samples_leaf=2, random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

print("Performing cross-validation...")
cv_scores = cross_val_score(clf, X_train_resampled, y_train_resampled, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")

print("Evaluating the model...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("Saving the model and vectorizer...")
model_data = {
    'classifier': clf,
    'vectorizer': tfidf,
    'label_encoder': le,
    'confidence_threshold': 0.3  # Increased threshold
}
pickle.dump(model_data, open('resume_classifier_v8.pkl', 'wb'))

print("Training complete. New model saved as resume_classifier_v8.pkl")