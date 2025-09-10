import streamlit as st
import PyPDF2, docx, string, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# Download stopwords quietly
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# ---------- Helper Functions ----------
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

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Resume Screening Bot", page_icon="📄", layout="wide")
st.header("📄 Resume Screening Bot")
st.markdown("Paste JD, upload resume (PDF/DOCX). Uses TF-IDF + Cosine Similarity to compute a match score.")

# Sidebar options
st.sidebar.markdown("### Options")
threshold = st.sidebar.slider("Highlight threshold (%)", min_value=0, max_value=100, value=50)
show_raw = st.sidebar.checkbox("Show raw extracted resume text", value=False)

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    job_description = st.text_area("Job Description", height=220, placeholder="Paste job description here...")
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    btn = st.button("🔍 Check Match")

with col2:
    st.markdown("### Tips")
    st.write("- Use clear JD bullets") 
    st.write("- PDF/Text resumes work best")
    st.write("- Increase JD detail for better match")
    st.markdown("---")
    st.markdown("### Legend")
    st.info("Green: match ≥ threshold\nYellow: close match\nBlue: below threshold")

# ---------- Main Logic ----------
if btn:
    if uploaded_file is None or job_description.strip() == "":
        st.error("⚠️ Please paste JD and upload a resume.")
    else:
        with st.spinner("Processing resume..."):
            if uploaded_file.name.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)

            jd_clean = clean_text(job_description)
            resume_clean = clean_text(resume_text)

            vec = TfidfVectorizer()
            tfidf = vec.fit_transform([jd_clean, resume_clean])

            score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]) * 100

        # Result display
        if score >= threshold:
            st.success(f"✅ Match Score: {score:.2f}%")
