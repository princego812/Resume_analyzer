"""
AI Resume Analyzer - single-file Streamlit app (BCA 307T, Natural Language Processing)
Run:  streamlit run app.py
"""
import io
import re
from collections import Counter

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer)
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# --------------------------------------------------------------------------
# Knowledge base: skills per career category
# --------------------------------------------------------------------------
SKILLS = {
    "Data Science / AI": [
        "python", "machine learning", "deep learning", "nlp", "natural language processing",
        "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "statistics",
        "data visualization", "matplotlib", "sql", "tf-idf", "nltk", "spacy", "transformers"],
    "Web Development": [
        "html", "css", "javascript", "react", "angular", "node.js", "express", "mongodb",
        "php", "bootstrap", "typescript", "rest api", "django", "flask", "mysql"],
    "Software Engineering": [
        "java", "c++", "c#", "python", "data structures", "algorithms", "oop", "git",
        "spring boot", "microservices", "docker", "testing", "sql", "linux", "agile"],
    "Networking / Security": [
        "network security", "cryptography", "firewall", "penetration testing", "wireshark",
        "tcp/ip", "vpn", "ids", "linux", "ethical hacking", "siem", "kali linux", "ssl", "malware analysis"],
}
ALL_SKILLS = sorted({s for v in SKILLS.values() for s in v})

SECTIONS = ["education", "experience", "projects", "skills", "certifications", "summary", "objective"]

# Small labelled corpus to train the Naive Bayes classifier (Unit IV)
TRAIN_TEXTS = {
    "Data Science / AI": [
        "built machine learning models in python using pandas numpy and scikit-learn",
        "trained deep learning neural networks with tensorflow and pytorch for image tasks",
        "nlp project sentiment analysis using nltk tf-idf and naive bayes classifier",
        "data analysis statistics visualization matplotlib sql dashboards",
        "text classification with transformers and spacy named entity recognition",
        "predictive modelling feature engineering regression clustering",
    ],
    "Web Development": [
        "developed responsive websites using html css javascript and bootstrap",
        "built single page applications with react node.js express and mongodb",
        "created rest api backend using django flask and mysql database",
        "frontend developer angular typescript ui ux responsive design",
        "full stack web application php mysql javascript",
        "designed web pages and integrated apis with react",
    ],
    "Software Engineering": [
        "developed backend services in java spring boot and microservices",
        "strong in data structures algorithms oop and c++ problem solving",
        "used git docker and agile methodology for software development and testing",
        "built desktop applications in c# and java with unit testing",
        "designed scalable software systems and code reviews in linux environment",
        "implemented design patterns object oriented programming sql database",
    ],
    "Networking / Security": [
        "configured firewall vpn and ids for network security monitoring",
        "performed penetration testing and ethical hacking using kali linux wireshark",
        "studied cryptography ssl tcp/ip protocols and malware analysis",
        "security analyst siem incident response vulnerability assessment",
        "network administration routers switches tcp/ip troubleshooting",
        "implemented encryption and secure authentication to prevent attacks",
    ],
}

stemmer = PorterStemmer()


# --------------------------------------------------------------------------
# Unit I + II : text as list of words, tokenization, stop words, stemming
# --------------------------------------------------------------------------
def tokenize(text):
    """Regex tokenizer that keeps tokens such as c++, c#, node.js, tcp/ip."""
    tokens = re.findall(r"[a-z][a-z0-9+#./-]*", text.lower())
    return [t.rstrip("./-") for t in tokens if t.rstrip("./-")]


def preprocess(text):
    """Full pipeline: lowercase -> tokenize -> stop-word removal -> stemming."""
    tokens = [t for t in tokenize(text) if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return [stemmer.stem(t) for t in tokens]


def clean_text(text):
    return " ".join(preprocess(text))


def word_frequency(text, top=10):
    words = [t for t in tokenize(text) if t not in ENGLISH_STOP_WORDS and len(t) > 2]
    return Counter(words).most_common(top)


# --------------------------------------------------------------------------
# Unit IV : information extraction (rule based NER)
# --------------------------------------------------------------------------
def extract_entities(text):
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone = re.search(r"(?:\+?\d{1,3}[\s-]?)?\d{10}|\+?\d[\d\s-]{9,13}\d", text)
    linkedin = re.search(r"linkedin\.com/[\w/-]+", text, re.I)
    github = re.search(r"github\.com/[\w-]+", text, re.I)
    first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
    name = first_line if len(first_line.split()) <= 4 else "Not found"
    years = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", text, re.I)
    degrees = re.findall(r"\b(b\.?tech|b\.?ca|m\.?ca|b\.?sc|m\.?sc|b\.?e|m\.?tech|mba|bachelor[^,\n]*|master[^,\n]*)",
                         text, re.I)
    return {
        "Name": name,
        "Email": email.group() if email else "Not found",
        "Phone": phone.group().strip() if phone else "Not found",
        "LinkedIn": linkedin.group() if linkedin else "Not found",
        "GitHub": github.group() if github else "Not found",
        "Experience (yrs mentioned)": max(map(float, years)) if years else 0,
        "Education": sorted({d.strip() for d in degrees}) or ["Not found"],
    }


def extract_skills(text):
    low = text.lower()
    found = []
    for skill in ALL_SKILLS:
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, low):
            found.append(skill)
    return found


def detect_sections(text):
    low = text.lower()
    return [s for s in SECTIONS if re.search(r"^\s*" + s, low, re.M)]


# --------------------------------------------------------------------------
# Unit III : vector space model, TF-IDF, n-grams, cosine similarity
# --------------------------------------------------------------------------
def tfidf_similarity(resume, jd):
    vec = TfidfVectorizer(ngram_range=(1, 2))
    m = vec.fit_transform([clean_text(resume), clean_text(jd)])
    return float(cosine_similarity(m[0], m[1])[0][0]), vec, m


def top_keywords(vec, matrix, row=1, n=8):
    names = vec.get_feature_names_out()
    scores = matrix[row].toarray()[0]
    idx = scores.argsort()[::-1][:n]
    return [names[i] for i in idx if scores[i] > 0]


def bow_demo(text):
    cv = CountVectorizer(ngram_range=(1, 1), max_features=8)
    cv.fit_transform([clean_text(text)])
    return list(cv.get_feature_names_out())


# --------------------------------------------------------------------------
# Unit IV : Naive Bayes classifier
# --------------------------------------------------------------------------
def build_classifier():
    X, y = [], []
    for label, docs in TRAIN_TEXTS.items():
        for d in docs:
            X.append(clean_text(d))
            y.append(label)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    clf = MultinomialNB(alpha=0.3)
    clf.fit(vec.fit_transform(X), y)
    return vec, clf


def classify(text, vec, clf):
    probs = clf.predict_proba(vec.transform([clean_text(text)]))[0]
    ranked = sorted(zip(clf.classes_, probs), key=lambda p: -p[1])
    return ranked


def evaluate_classifier():
    """Train/test split + accuracy (Unit III: training and test sets, evaluation)."""
    X, y = [], []
    for label, docs in TRAIN_TEXTS.items():
        for d in docs:
            X.append(clean_text(d))
            y.append(label)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
    vec = TfidfVectorizer(ngram_range=(1, 2))
    clf = MultinomialNB(alpha=0.3).fit(vec.fit_transform(Xtr), ytr)
    return accuracy_score(yte, clf.predict(vec.transform(Xte))), len(Xtr), len(Xte)


# --------------------------------------------------------------------------
# Scoring and report
# --------------------------------------------------------------------------
def analyze(resume, jd):
    r_skills, j_skills = set(extract_skills(resume)), set(extract_skills(jd))
    matched = sorted(r_skills & j_skills)
    missing = sorted(j_skills - r_skills)
    skill_score = len(matched) / len(j_skills) if j_skills else 0
    sim, vec, mat = tfidf_similarity(resume, jd)
    sections = detect_sections(resume)
    section_score = min(len(set(sections) & {"education", "experience", "projects", "skills"}) / 4, 1)

    final = 100 * (0.5 * skill_score + 0.3 * min(sim * 2, 1) + 0.2 * section_score)
    return {
        "score": round(final, 1), "skill_score": skill_score, "similarity": sim,
        "matched": matched, "missing": missing, "sections": sections,
        "resume_skills": sorted(r_skills), "jd_keywords": top_keywords(vec, mat),
    }


def verdict(score):
    if score >= 75:
        return "Strong match"
    if score >= 50:
        return "Moderate match - improve missing skills"
    return "Weak match - resume needs tailoring"

import pandas as pd
import streamlit as st


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
    return build_classifier()


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

    res = analyze(resume, jd)
    ent = extract_entities(resume)
    vec, clf = get_classifier()
    ranked = classify(resume, vec, clf)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall ATS score", f"{res['score']}/100")
    c2.metric("Skill coverage", f"{res['skill_score'] * 100:.0f}%")
    c3.metric("TF-IDF similarity", f"{res['similarity']:.2f}")
    st.progress(min(int(res["score"]), 100))
    verdict_text = verdict(res["score"])
    (st.success if res["score"] >= 75 else st.warning if res["score"] >= 50 else st.error)(verdict_text)

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
            acc, ntr, nte = evaluate_classifier()
            st.write(f"Accuracy: **{acc * 100:.0f}%** (train={ntr}, test={nte}). Small demo dataset, so treat as indicative only.")

    with tab4:
        st.markdown("**Pre-processing pipeline** (tokenize → stop-word removal → stemming)")
        st.code(" ".join(preprocess(resume)[:40]) or "-")
        st.markdown("**Top words (Unit I word frequency)**")
        wf = pd.DataFrame(word_frequency(resume, 10), columns=["Word", "Count"])
        st.bar_chart(wf.set_index("Word"))
        st.markdown("**Bag-of-Words terms:** " + ", ".join(bow_demo(resume)))
        st.markdown("**Top job-description keywords (TF-IDF):** " + ", ".join(res["jd_keywords"]))
