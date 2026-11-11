"""
AI Resume Analyzer - Streamlit app (BCA 307T, Natural Language Processing)

Run:  streamlit run app.py
Needs resume_analyzer.py in the same folder (it holds the NLP logic).
"""
import io

import pandas as pd
import streamlit as st

import resume_analyzer as ra

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

SAMPLE_RESUME = """Rahul Sharma
Email: rahul.sharma@example.com | Phone: 9876543210
linkedin.com/in/rahulsharma | github.com/rahul-s

Summary
BCA student with 1 year of project experience in Python and machine learning.

Education
BCA, Computer Applications, 2026

Skills
Python, SQL, pandas, numpy, scikit-learn, NLTK, HTML, CSS, Git

Projects
Sentiment Analysis of Movie Reviews - used tokenization, TF-IDF and Naive Bayes classifier with NLTK and scikit-learn.
Portfolio Website - built a responsive website using HTML, CSS and JavaScript.

Experience
Data analysis intern - cleaned datasets with pandas and created data visualization dashboards using matplotlib.
"""

SAMPLE_JD = """NLP / Machine Learning Intern

We are looking for an intern with knowledge of Python, machine learning and natural language processing.
Required skills: python, nlp, nltk, spacy, tf-idf, scikit-learn, pandas, sql, git, deep learning, pytorch.
Responsibilities: build text classification models, perform named entity recognition and data visualization,
and present findings to the team.
"""


# ---------------------------------------------------------------- helpers
def read_upload(file):
    """Return text from an uploaded .txt, .pdf or .docx file."""
    name = file.name.lower()
    data = file.read()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx
        d = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in d.paragraphs)
    return data.decode("utf-8", errors="ignore")


def get_text(label, key, sample):
    mode = st.radio(f"{label} input", ["Paste text", "Upload file"], horizontal=True, key=key + "_mode")
    if mode == "Upload file":
        f = st.file_uploader(f"Upload {label.lower()} (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"], key=key + "_file")
        if f:
            text = read_upload(f)
            if not text.strip():
                st.warning("No text could be read from this file (scanned PDFs need OCR).")
            return text
        return ""
    return st.text_area(label, value=sample, height=320, key=key + "_text")


@st.cache_resource
def get_classifier():
    return ra.build_classifier()


# ---------------------------------------------------------------- layout
st.title("📄 AI Resume Analyzer")
st.caption("NLP mini project · BCA 307T · tokenization, TF-IDF, cosine similarity, Naive Bayes")

with st.sidebar:
    st.header("About")
    st.write("Compares a resume with a job description and scores the match using classic NLP techniques.")
    st.markdown(
        "- **Unit I**: words, frequency, ethics\n"
        "- **Unit II**: tokenize, stop words, stemming\n"
        "- **Unit III**: BoW, TF-IDF, n-grams, cosine similarity\n"
        "- **Unit IV**: information extraction, Naive Bayes"
    )
    st.info("Name, gender and age are never used in scoring. Use this tool to assist a human, not replace one.")

left, right = st.columns(2)
with left:
    st.subheader("Resume")
    resume = get_text("Resume", "resume", SAMPLE_RESUME)
with right:
    st.subheader("Job description")
    jd = get_text("Job description", "jd", SAMPLE_JD)

if st.button("Analyze", type="primary", use_container_width=True):
    if not resume.strip() or not jd.strip():
        st.error("Please provide both a resume and a job description.")
        st.stop()

    res = ra.analyze(resume, jd)
    ent = ra.extract_entities(resume)
    vec, clf = get_classifier()
    ranked = ra.classify(resume, vec, clf)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall ATS score", f"{res['score']}/100")
    c2.metric("Skill coverage", f"{res['skill_score'] * 100:.0f}%")
    c3.metric("TF-IDF similarity", f"{res['similarity']:.2f}")
    st.progress(min(int(res["score"]), 100))
    verdict = ra.verdict(res["score"])
    (st.success if res["score"] >= 75 else st.warning if res["score"] >= 50 else st.error)(verdict)

    tab1, tab2, tab3, tab4 = st.tabs(["Skills match", "Candidate details", "Career category", "NLP internals"])

    with tab1:
        a, b = st.columns(2)
        a.markdown("**✅ Matched skills**")
        a.write(", ".join(res["matched"]) or "None")
        b.markdown("**❌ Missing skills**")
        b.write(", ".join(res["missing"]) or "None 🎉")
        st.markdown("**All skills found in resume**")
        st.write(", ".join(res["resume_skills"]) or "None")
        st.markdown("**Sections detected:** " + (", ".join(res["sections"]) or "None"))
        missing_sections = [s for s in ["education", "experience", "projects", "skills"] if s not in res["sections"]]
        if missing_sections:
            st.warning("Consider adding sections: " + ", ".join(missing_sections))

    with tab2:
        st.table(pd.DataFrame(
            [(k, ", ".join(v) if isinstance(v, list) else str(v)) for k, v in ent.items()],
            columns=["Field", "Value"]).set_index("Field"))

    with tab3:
        df = pd.DataFrame(ranked, columns=["Category", "Probability"])
        st.bar_chart(df.set_index("Category"))
        st.write(f"Predicted category: **{ranked[0][0]}** ({ranked[0][1] * 100:.1f}%)")
        if st.checkbox("Show classifier evaluation (train/test split)"):
            acc, ntr, nte = ra.evaluate_classifier()
            st.write(f"Accuracy: **{acc * 100:.0f}%** (train={ntr}, test={nte}). Small demo dataset, so treat as indicative only.")

    with tab4:
        st.markdown("**Pre-processing pipeline** (tokenize → stop-word removal → stemming)")
        st.code(" ".join(ra.preprocess(resume)[:40]) or "-")
        st.markdown("**Top words (Unit I word frequency)**")
        wf = pd.DataFrame(ra.word_frequency(resume, 10), columns=["Word", "Count"])
        st.bar_chart(wf.set_index("Word"))
        st.markdown("**Bag-of-Words terms:** " + ", ".join(ra.bow_demo(resume)))
        st.markdown("**Top job-description keywords (TF-IDF):** " + ", ".join(res["jd_keywords"]))
