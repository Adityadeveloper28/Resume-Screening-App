import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from PyPDF2 import PdfReader
import io
import base64
import time
from streamlit import components

# Set page config at the very beginning of the script
st.set_page_config(page_title="Resume Screening App", layout="wide", initial_sidebar_state="expanded")

# Download necessary NLTK data silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load the updated model
@st.cache_resource
def load_model():
    with open('resume_classifier_v8.pkl', 'rb') as f:
        return pickle.load(f)

model_data = load_model()

# Unpacking the model components
clf = model_data['classifier']
tfidf = model_data['vectorizer']
le = model_data['label_encoder']
CONFIDENCE_THRESHOLD = model_data['confidence_threshold']

# Helper function to clean resume text
def clean_resume(resume_text):
    resume_text = re.sub(r'http\S+\s*', ' ', resume_text)
    resume_text = re.sub('RT|cc', ' ', resume_text)
    resume_text = re.sub(r'#\S+', '', resume_text)
    resume_text = re.sub(r'@\S+', '  ', resume_text)
    resume_text = re.sub(r'[^\w\s]', '', resume_text)
    resume_text = re.sub(r'\d+', '', resume_text)
    resume_text = re.sub(r'\s+', ' ', resume_text).strip().lower()
    return resume_text

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    pdf_reader = PdfReader(io.BytesIO(file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

# Map category indices to their corresponding labels
category_mapping = dict(zip(range(len(le.classes_)), le.classes_))

# Function to get top N categories based on prediction probabilities
def get_top_categories(prediction_proba, top_n=5):
    top_indices = prediction_proba.argsort()[-top_n:][::-1]
    top_probs = prediction_proba[top_indices]
    top_categories = [category_mapping.get(idx, "Unknown") for idx in top_indices]
    return list(zip(top_categories, top_probs))

# Function to get feature importance for a category
def get_feature_importance(cleaned_resume, category):
    try:
        feature_vector = tfidf.transform([cleaned_resume])
        feature_names = tfidf.get_feature_names_out()
        category_index = le.transform([category])[0]
        
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = clf.coef_.ravel()
        else:
            raise AttributeError("Classifier doesn't have feature_importances_ or coef_ attribute")
        
        if importances.ndim > 1:
            importances = importances[category_index]
        
        non_zero_features = feature_vector.nonzero()[1]
        feature_importances = [(feature_names[i], importances[i]) for i in non_zero_features]
        feature_importances.sort(key=lambda x: abs(x[1]), reverse=True)
        return feature_importances[:10]
    except Exception as e:
        st.error(f"An error occurred while getting feature importance: {str(e)}")
        return []

# Function to display PDF
def display_pdf(file):
    # Encode PDF file to base64
    base64_pdf = base64.b64encode(file).decode('utf-8')
    
    # Embed PDF viewer
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    
    # Display the PDF viewer
    st.markdown(pdf_display, unsafe_allow_html=True)

# Try to import plotly, if not available, set a flag
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly is not installed. Some visualizations will not be available. Please install plotly using 'pip install plotly'.")

# Main function to run the Streamlit app
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("Resume Screening Tool")
    uploaded_file = st.sidebar.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        file_contents = uploaded_file.read()

        with st.spinner('Processing resume...'):
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(file_contents)
            else:
                try:
                    resume_text = file_contents.decode('utf-8')
                except UnicodeDecodeError:
                    resume_text = file_contents.decode('latin-1')

            # Clean and process the uploaded resume
            processed_resume = clean_resume(resume_text)

        st.success('Resume processed successfully!')

        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Resume")
            if uploaded_file.type == "application/pdf":
                display_pdf(file_contents)
            else:
                st.text_area("Original Text", resume_text, height=400)
        
        with col2:
            st.subheader("Processed Resume")
            st.text_area("Processed Text", processed_resume[:1000] + "..." if len(processed_resume) > 1000 else processed_resume, height=400)
        
        # Download button for processed resume
        st.download_button(
            label="Download Full Processed Resume Text",
            data=processed_resume,
            file_name="processed_resume.txt",
            mime="text/plain"
        )

        # Convert processed resume to input features using TF-IDF vectorizer
        input_features = tfidf.transform([processed_resume])

        # Predict category probabilities and normalize them
        prediction_proba = clf.predict_proba(input_features)[0]
        prediction_proba = normalize(prediction_proba.reshape(1, -1), norm='l1')[0]

        top_categories = get_top_categories(prediction_proba)

        st.subheader("Top 5 Matching Job Categories")
        
        categories, probabilities = zip(*top_categories)
        
        if PLOTLY_AVAILABLE:
            # Interactive bar chart
            fig = go.Figure(data=[go.Bar(x=categories, y=[p*100 for p in probabilities])])
            fig.update_layout(title='Top 5 Job Categories - Probabilities', yaxis_title='Probability (%)')
            st.plotly_chart(fig)
        else:
            # Fallback to matplotlib
            fig, ax = plt.subplots()
            ax.bar(categories, [p*100 for p in probabilities])
            ax.set_ylabel('Probability (%)')
            ax.set_title('Top 5 Job Categories - Probabilities')
            plt.xticks(rotation=45)
            st.pyplot(fig)

        # Display category predictions and confidence
        for category, prob in top_categories:
            percentage = prob * 100
            confidence_label = "High" if prob >= CONFIDENCE_THRESHOLD else "Low"
            st.write(f"{category}: {percentage:.2f}% (Confidence: {confidence_label})")
            
            with st.expander(f"See top features for {category}"):
                feature_importances = get_feature_importance(processed_resume, category)
                if feature_importances:
                    for feature, importance in feature_importances:
                        st.write(f"- {feature}: {importance:.4f}")
                else:
                    st.write("No feature importance information available for this category.")

        # Show the model's confidence threshold
        st.subheader("Confidence Threshold")
        st.write(f"The model's confidence threshold is set to {CONFIDENCE_THRESHOLD*100}%.")
        st.write("Predictions below this threshold are marked as 'Low' confidence.")

        # Analyze the distribution of probabilities across all categories
        with st.expander("See distribution across all categories"):
            st.write("Distribution of probabilities across all categories:")
            all_categories = [category_mapping[i] for i in range(len(category_mapping))]
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=[go.Bar(x=all_categories, y=prediction_proba * 100)])
                fig.update_layout(title='All Categories - Probabilities', xaxis_tickangle=-45, yaxis_title='Probability (%)')
                st.plotly_chart(fig)
            else:
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.bar(all_categories, prediction_proba * 100)
                ax.set_ylabel('Probability (%)')
                ax.set_title('All Categories - Probabilities')
                plt.xticks(rotation=90)
                st.pyplot(fig)

        # Display the top N-grams found in the resume
        st.subheader("Top N-grams in Resume")
        feature_vector = tfidf.transform([processed_resume])
        feature_names = tfidf.get_feature_names_out()
        sorted_indices = feature_vector.data.argsort()[-10:][::-1]
        top_ngrams = [feature_names[feature_vector.indices[i]] for i in sorted_indices]
        st.write(", ".join(top_ngrams))

if __name__ == "__main__":
    main()