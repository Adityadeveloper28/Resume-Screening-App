import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from PyPDF2 import PdfReader
import io
import base64

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Loading models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidfd = pickle.load(open('tfidf.pkl', 'rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

def extract_text_from_pdf(file):
    pdf_reader = PdfReader(io.BytesIO(file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""  # Handle cases where extract_text might return None
    return text

# Category mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
    18: "Operations Manager", 6: "Data Science", 22: "Sales", 16: "Mechanical Engineer",
    1: "Arts", 7: "Database", 11: "Electrical Engineering", 14: "Health and fitness",
    19: "PMO", 4: "Business Analyst", 9: "DotNet Developer", 2: "Automation Testing",
    17: "Network Security Engineer", 21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"
}

def get_top_categories(prediction_proba, top_n=5):
    top_indices = prediction_proba.argsort()[-top_n:][::-1]
    top_probs = prediction_proba[top_indices]
    top_categories = [category_mapping.get(idx, "Unknown") for idx in top_indices]
    return list(zip(top_categories, top_probs))

def display_pdf(file):
    base64_pdf = base64.b64encode(file).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    return pdf_display  # Return the PDF display string

def render_bar_chart_matplotlib(categories, probabilities):
    fig, ax = plt.subplots()

    # Convert probabilities to percentages
    percentages = [prob * 100 for prob in probabilities]

    # Create bar chart with categories on the x-axis
    ax.bar(categories, percentages, color='blue')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Top 5 Job Categories - Probabilities')
    ax.set_xticklabels(categories, rotation=45, ha='right')  # Rotate x-axis labels for better readability

    # Display the plot
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Resume Screening App", layout="wide")
    
    # Tailwind CSS CDN
    st.markdown("""<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">""", unsafe_allow_html=True)
    
    # Custom CSS for Streamlit components
    st.markdown("""<style>
    .stApp { background-color: #f3f4f6; }
    .stProgress > div > div > div { background-color: #3b82f6; }
    .stFileUploader > div > div {
        background-color: #e5e7eb !important;
        padding: 0.5rem !important;
        border-radius: 0.375rem;
    }
    .upload-btn { background-color: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 0.375rem; font-weight: 600; cursor: pointer; transition: background-color 0.3s; }
    .upload-btn:hover { background-color: #2563eb; }
    .stFileUploader > div > div:nth-child(2) { color: black !important; font-weight: 600; margin-top: 0.5rem; }
    .st-emotion-cache-uef7qa p { color: black; }
    .st-emotion-cache-12xsiil { display: flex; -webkit-box-align: center; color: black; align-items: center; margin-bottom: 0.25rem; }
    </style>
    """, unsafe_allow_html=True)

    # App layout with improved Tailwind CSS
    st.markdown("""<div class="container mx-auto"><div class="text-center mb-12"><h1 class="text-5xl font-extrabold text-blue-600 mb-4">Resume Screening App</h1><p class="text-xl text-gray-600">Upload a resume to predict the top 5 matching job categories</p></div><div class="">""", unsafe_allow_html=True)

    # File uploader inside the container
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'], label_visibility="collapsed")

    if uploaded_file is not None:
        file_contents = uploaded_file.read()

        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(file_contents)
            st.markdown(f'<div class="bg-white shadow-lg rounded-lg p-8 mb-8"><h2 class="text-3xl font-bold text-blue-600 mb-6">Uploaded PDF</h2>{display_pdf(file_contents)}</div>', unsafe_allow_html=True)
        else:
            try:
                resume_text = file_contents.decode('utf-8')
            except UnicodeDecodeError:
                resume_text = file_contents.decode('latin-1')

        cleaned_resume = clean_resume(resume_text)
        input_features = tfidfd.transform([cleaned_resume])

        prediction_proba = clf.predict_proba(input_features)[0]
        prediction_proba = normalize(prediction_proba.reshape(1, -1), norm='l1')[0]

        top_categories = get_top_categories(prediction_proba, top_n=5)

        # Cleaned Resume Preview Section
        st.markdown(f'''
        <div class="bg-white shadow-lg rounded-lg p-8">
            <h2 class="text-3xl font-bold text-blue-600 mb-6">Cleaned Resume Preview</h2>
            <div class="bg-gray-100 p-6 rounded-lg">
                <p class="text-gray-800 leading-relaxed">{cleaned_resume}...</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)

        st.markdown(f'<div class="bg-white shadow-lg rounded-lg p-8 mb-8"><h2 class="text-3xl font-bold text-blue-600 mb-6">Top 5 Matching Job Categories</h2>', unsafe_allow_html=True)

        categories = []
        probabilities = []
        
        for category, prob in top_categories:
            percentage = prob * 100
            categories.append(category)
            probabilities.append(prob)
            st.markdown(f'<div class="mb-6"><div class="flex justify-between items-center mb-2"><span class="text-xl font-semibold text-gray-800">{category}</span><span class="text-lg font-medium text-blue-600">{percentage:.2f}%</span></div><div class="w-full bg-gray-200 rounded-full h-4"><div class="bg-blue-600 h-4 rounded-full" style="width: {percentage}%"></div></div></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Render the bar chart using Matplotlib
        render_bar_chart_matplotlib(categories, probabilities)

if __name__ == "__main__":
    main()

