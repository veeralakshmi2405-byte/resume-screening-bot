import streamlit as st
import PyPDF2, docx, string, nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io

# For OCR
from pdf2image import convert_from_path
import pytesseract
from pytesseract import Output

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

def extract_text_from_scanned_pdf(uploaded_file):
    # Convert PDF pages to images + OCR
    images = convert_from_path(uploaded_file, dpi=300)
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img, lang='eng')
    return text

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
threshold = st.sidebar.slider("Highlight threshold (%)", 0, 100, 50)
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
    if not uploaded_file or not job_description.strip():
        st.error("⚠️ Please paste JD and upload a resume.")
    else:
        with st.spinner("Processing resume..."):
            # Extract resume text
            if uploaded_file.name.lower().endswith(".pdf"):
                resume_text = extract_text_from_pdf(uploaded_file)
                if not resume_text.strip():
                    st.info("Text extraction returned empty — trying OCR fallback.")
                    uploaded_file.seek(0)
                    resume_text = extract_text_from_scanned_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)

            # Debug info (optional)
            st.write("Resume extracted text length:", len(resume_text))

            # Clean text
            jd_clean = clean_text(job_description)
            resume_clean = clean_text(resume_text)

            st.write("JD words:", len(jd_clean.split()), "| Resume words:", len(resume_clean.split()))

            # Compute similarity
            vec = TfidfVectorizer()
            tfidf = vec.fit_transform([jd_clean, resume_clean])
            try:
                score = float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]) * 100
            except Exception as e:
                st.error(f"Error computing similarity: {e}")
                st.stop()

        # Show result
        if score >= threshold:
            st.success(f"✅ Match Score: {score:.2f}%")
        elif score >= threshold * 0.7:
            st.warning(f"⚠️ Match Score: {score:.2f}% (close)")
        else:
            st.info(f"ℹ️ Match Score: {score:.2f}%")

        if show_raw:
            st.markdown("**Extracted Resume Text (first 500 chars):**")
            st.text_area("Resume Text", value=resume_text[:500], height=200)

        # Download button
        result_txt = f"Match Score: {score:.2f}%\nResume file: {uploaded_file.name}\n"
        buf = io.BytesIO(result_txt.encode("utf-8"))
        st.download_button("⬇️ Download Result", data=buf, file_name="resume_match.txt", mime="text/plain")
