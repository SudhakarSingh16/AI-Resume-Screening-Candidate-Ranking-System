import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    """Extracts text from a given PDF file."""
    try:
        pdf = PdfReader(file)
        return "\n".join(filter(None, (page.extract_text() for page in pdf.pages))) or "No extractable text found."
    except Exception as e:
        return f"Error extracting text: {e}"

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    """Computes cosine similarity scores for ranking resumes."""
    try:
        documents = [job_description] + resumes
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)  # Uses sparse matrices for efficiency
        cosine_similarities = cosine_similarity(vectors[0], vectors[1:]).flatten()
        return cosine_similarities
    except Exception as e:
        return f"Error computing similarity: {e}"

# Streamlit UI Configuration
st.set_page_config(page_title="AI Resume Ranker", layout="centered")
st.title("📄 AI Resume Screening & Candidate Ranking System")

# Job description input
job_description = st.text_area("📌 Enter the job description:")

# File uploader for resumes
uploaded_files = st.file_uploader("📂 Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

# Button to analyze resumes
if st.button("🔍 Analyze Resumes"):
    if not job_description:
        st.warning("⚠ Please enter a job description.")
    elif not uploaded_files:
        st.warning("⚠ Please upload at least one resume.")
    else:
        st.header("📊 Ranking Resumes")
        
        # Extract text from uploaded resumes
        with st.spinner("Extracting text from resumes..."):
            resumes = [extract_text_from_pdf(file) for file in uploaded_files]

        # Compute rankings
        scores = rank_resumes(job_description, resumes)

        if isinstance(scores, str):  # Check if error occurred
            st.error(scores)
        else:
            # Prepare results
            results = pd.DataFrame({
                "Resume": [file.name for file in uploaded_files],
                "Score (%)": (scores * 100).round(2)
            }).sort_values(by="Score (%)", ascending=False)

            st.success("✅ Ranking complete!")
            st.write(results)

            # Highlight top candidate
            if not results.empty:
                top_candidate = results.iloc[0]
                st.subheader(f"🏆 Top Candidate: {top_candidate['Resume']} (Score: {top_candidate['Score (%)']}%)")

# Sidebar Developer Information
st.sidebar.markdown("### 👨‍💻 Developed by *Sudhakar Singh*")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/sudhakar-singh-7737a72a6/)")
st.sidebar.markdown("[🐙 GitHub](https://github.com/sudhakar-singh)")