import streamlit as st
import PyPDF2, docx, string, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return " ".join([p.text for p in doc.paragraphs])

def clean_text(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.set_page_config(page_title="Resume Screening Bot", page_icon="📄", layout="centered")
st.title("📄 Resume Screening Bot")
st.markdown("Paste JD, upload resume (pdf/docx) — TF-IDF + Cosine Similarity gives a match score.")

job_description = st.text_area("Job Description", height=200)
uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf","docx"])

if st.button("🔍 Check Match"):
    if uploaded_file is None or job_description.strip()=="":
        st.error("Please paste JD and upload a resume.")
    else:
        if uploaded_file.name.lower().endswith(".pdf"):
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = extract_text_from_docx(uploaded_file)

        jd_clean = clean_text(job_description)
        resume_clean = clean_text(resume_text)

        vec = TfidfVectorizer()
        tfidf = vec.fit_transform([jd_clean, resume_clean])
        score
