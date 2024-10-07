import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
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
df = pd.read_csv('UpdatedResumeDataSet_v2.csv')
df['cleaned_resume'] = df['Resume'].apply(clean_resume)

print("\nCategory Distribution:")
category_counts = df['Category'].value_counts()
for category, count in category_counts.items():
    print(f"{category}: {count}")

custom_stop_words = ['experience', 'skill', 'work', 'project', 'developed', 'implemented']
stop_words = list(set(stopwords.words('english')).union(custom_stop_words))

print("\nExtracting features...")
tfidf = TfidfVectorizer(
    stop_words=stop_words,
    ngram_range=(1, 2),
    max_features=5000,
    max_df=0.5
)
features = tfidf.fit_transform(df['cleaned_resume'])

print("\nTop 10 features for each category:")
feature_names = np.array(tfidf.get_feature_names_out())
for category in df['Category'].unique():
    category_docs = df[df['Category'] == category]['cleaned_resume']
    category_features = tfidf.transform(category_docs)
    top_feature_indices = category_features.sum(axis=0).argsort()[0, -10:]
    top_features = feature_names[top_feature_indices.tolist()[0]]
    print(f"\n{category}:")
    print(", ".join(top_features))

print("\nAnalysis complete. Review the output to identify any changes in category distribution or key features.")