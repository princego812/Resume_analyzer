
"""
AI Resume Analyzer
NLP Mini Project - BCA 307T
Run: streamlit run app.py
"""

import io
import re
from collections import Counter

import pandas as pd
import streamlit as st

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer
)
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# -----------------------------------------
# 1. SKILLS DATABASE
# -----------------------------------------

SKILLS = {
    "Data Science / AI": [
        "python", "machine learning", "deep learning",
        "nlp", "natural language processing",
        "tensorflow", "pytorch", "scikit-learn",
        "pandas", "numpy", "statistics",
        "data visualization", "matplotlib",
        "sql", "tf-idf", "nltk", "spacy", "transformers"
    ],

    "Web Development": [
        "html", "css", "javascript", "react",
        "angular", "node.js", "express", "mongodb",
        "php", "bootstrap", "typescript",
        "rest api", "django", "flask", "mysql"
    ],

    "Software Engineering": [
        "java", "c++", "c#", "python",
        "data structures", "algorithms", "oop",
        "git", "spring boot", "microservices",
        "docker", "testing", "sql", "linux", "agile"
    ],

    "Networking / Security": [
        "network security", "cryptography",
        "firewall", "penetration testing",
        "wireshark", "tcp/ip", "vpn", "ids",
        "linux", "ethical hacking", "siem",
        "kali linux", "ssl", "malware analysis"
    ]
}

ALL_SKILLS = sorted({
    skill
    for skills in SKILLS.values()
    for skill in skills
})

SECTIONS = [
    "education", "experience", "projects",
    "skills", "certifications", "summary", "objective"
]


# -----------------------------------------
# 2. TRAINING DATASET
# -----------------------------------------

TRAIN_TEXTS = {
    "Data Science / AI": [
        "machine learning python pandas numpy scikit-learn",
        "deep learning tensorflow pytorch neural networks",
        "nlp sentiment analysis nltk tf-idf naive bayes",
        "data analysis statistics visualization matplotlib sql",
        "text classification transformers spacy",
        "predictive modelling feature engineering regression"
    ],

    "Web Development": [
        "html css javascript bootstrap responsive websites",
        "react node.js express mongodb applications",
        "django flask mysql backend rest api",
        "angular typescript frontend development",
        "php mysql javascript full stack",
        "react api integration web pages"
    ],

    "Software Engineering": [
        "java spring boot backend microservices",
        "data structures algorithms oop c++",
        "git docker agile software testing",
        "c# java desktop applications unit testing",
        "software systems linux code reviews",
        "design patterns object oriented programming sql"
    ],

    "Networking / Security": [
        "firewall vpn ids network security monitoring",
        "penetration testing ethical hacking kali linux",
        "cryptography ssl tcp ip malware analysis",
        "security analyst siem incident response",
        "network administration routers switches",
        "encryption secure authentication cyber attacks"
    ]
}


stemmer = PorterStemmer()


# -----------------------------------------
# 3. TEXT PREPROCESSING
# -----------------------------------------

def tokenize(text):
    tokens = re.findall(
        r"[a-z][a-z0-9+#./-]*",
        text.lower()
    )

    return [
        token.rstrip("./-")
        for token in tokens
        if token.rstrip("./-")
    ]


def preprocess(text):
    tokens = [
        word for word in tokenize(text)
        if word not in ENGLISH_STOP_WORDS
        and len(word) > 1
    ]

    return [stemmer.stem(word) for word in tokens]


def clean_text(text):
    return " ".join(preprocess(text))


def word_frequency(text, top=10):
    words = [
        word for word in tokenize(text)
        if word not in ENGLISH_STOP_WORDS
        and len(word) > 2
    ]

    return Counter(words).most_common(top)


# -----------------------------------------
# 4. INFORMATION EXTRACTION
# -----------------------------------------

def extract_entities(text):

    email = re.search(
        r"[\w.+-]+@[\w-]+\.[\w.-]+",
        text
    )

    phone = re.search(
        r"(?:\+?\d{1,3}[\s-]?)?\d{10}|\+?\d[\d\s-]{9,13}\d",
        text
    )

    linkedin = re.search(
        r"linkedin\.com/[\w/-]+",
        text,
        re.I
    )

    github = re.search(
        r"github\.com/[\w-]+",
        text,
        re.I
    )

    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        ""
    )

    name = (
        first_line
        if len(first_line.split()) <= 4
        else "Not found"
    )

    years = re.findall(
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
        text,
        re.I
    )

    degrees = re.findall(
        r"\b(b\.?tech|b\.?ca|m\.?ca|b\.?sc|m\.?sc|b\.?e|m\.?tech|mba|bachelor[^,\n]*|master[^,\n]*)",
        text,
        re.I
    )

    return {
        "Name": name,
        "Email": email.group() if email else "Not found",
        "Phone": phone.group().strip() if phone else "Not found",
        "LinkedIn": linkedin.group() if linkedin else "Not found",
        "GitHub": github.group() if github else "Not found",
        "Experience": max(map(float, years)) if years else 0,
        "Education": sorted(set(degrees)) or ["Not found"]
    }


# -----------------------------------------
# 5. SKILL EXTRACTION
# -----------------------------------------

def extract_skills(text):

    text = text.lower()
    found = []

    for skill in ALL_SKILLS:

        pattern = (
            r"(?<![a-z0-9])"
            + re.escape(skill)
            + r"(?![a-z0-9])"
        )

        if re.search(pattern, text):
            found.append(skill)

    return found


def detect_sections(text):

    text = text.lower()

    return [
        section
        for section in SECTIONS
        if re.search(r"^\s*" + section, text, re.M)
    ]


# -----------------------------------------
# 6. TF-IDF AND COSINE SIMILARITY
# -----------------------------------------

def tfidf_similarity(resume, job_description):

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    matrix = vectorizer.fit_transform([
        clean_text(resume),
        clean_text(job_description)
    ])

    similarity = cosine_similarity(
        matrix[0],
        matrix[1]
    )[0][0]

    return float(similarity), vectorizer, matrix


def top_keywords(vectorizer, matrix, row=1, n=8):

    names = vectorizer.get_feature_names_out()
    scores = matrix[row].toarray()[0]

    indexes = scores.argsort()[::-1][:n]

    return [
        names[i]
        for i in indexes
        if scores[i] > 0
    ]


# -----------------------------------------
# 7. BAG OF WORDS
# -----------------------------------------

def bow_demo(text):

    vectorizer = CountVectorizer(
        ngram_range=(1, 1),
        max_features=8
    )

    vectorizer.fit_transform([clean_text(text)])

    return list(vectorizer.get_feature_names_out())


# -----------------------------------------
# 8. NAIVE BAYES CLASSIFIER
# -----------------------------------------

def build_classifier():

    X = []
    y = []

    for label, documents in TRAIN_TEXTS.items():

        for document in documents:
            X.append(clean_text(document))
            y.append(label)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    classifier = MultinomialNB(alpha=0.3)

    classifier.fit(
        vectorizer.fit_transform(X),
        y
    )

    return vectorizer, classifier


def classify(text, vectorizer, classifier):

    probabilities = classifier.predict_proba(
        vectorizer.transform([clean_text(text)])
    )[0]

    ranked = sorted(
        zip(classifier.classes_, probabilities),
        key=lambda item: -item[1]
    )

    return ranked


def evaluate_classifier():

    X = []
    y = []

    for label, documents in TRAIN_TEXTS.items():

        for document in documents:
            X.append(clean_text(document))
            y.append(label)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=7,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2)
    )

    classifier = MultinomialNB(alpha=0.3)

    classifier.fit(
        vectorizer.fit_transform(X_train),
        y_train
    )

    predictions = classifier.predict(
        vectorizer.transform(X_test)
    )

    accuracy = accuracy_score(y_test, predictions)

    return accuracy, len(X_train), len(X_test)


# -----------------------------------------
# 9. RESUME ANALYSIS
# -----------------------------------------

def analyze(resume, job_description):

    resume_skills = set(extract_skills(resume))
    job_skills = set(extract_skills(job_description))

    matched = sorted(resume_skills & job_skills)
    missing = sorted(job_skills - resume_skills)

    skill_score = (
        len(matched) / len(job_skills)
        if job_skills else 0
    )

    similarity, vectorizer, matrix = tfidf_similarity(
        resume,
        job_description
    )

    sections = detect_sections(resume)

    section_score = min(
        len(set(sections) & {
            "education", "experience", "projects", "skills"
        }) / 4,
        1
    )

    final_score = 100 * (
        0.5 * skill_score
        + 0.3 * min(similarity * 2, 1)
        + 0.2 * section_score
    )

    return {
        "score": round(final_score, 1),
        "skill_score": skill_score,
        "similarity": similarity,
        "matched": matched,
        "missing": missing,
        "sections": sections,
        "resume_skills": sorted(resume_skills),
        "jd_keywords": top_keywords(vectorizer, matrix)
    }


def get_verdict(score):

    if score >= 75:
        return "Strong match"

    if score >= 50:
        return "Moderate match - improve missing skills"

    return "Weak match - resume needs tailoring"


# -----------------------------------------
# 10. SAMPLE RESUME AND JOB DESCRIPTION
# -----------------------------------------

SAMPLE_RESUME = """
Rahul Sharma

Email: rahul.sharma@example.com
Phone: 9876543210

linkedin.com/in/rahulsharma
github.com/rahul-s

Summary

BCA student with 1 year of project experience
in Python and machine learning.

Education

BCA, Computer Applications, 2026

Skills

Python, SQL, pandas, numpy, scikit-learn,
NLTK, HTML, CSS, Git

Projects

Sentiment Analysis of Movie Reviews.
Used tokenization, TF-IDF and Naive Bayes.

Portfolio Website.
Built using HTML, CSS and JavaScript.

Experience

Data analysis intern.
Cleaned datasets using pandas.
Created dashboards using matplotlib.
"""


SAMPLE_JD = """
NLP / Machine Learning Intern

Required skills:

Python, NLP, NLTK, spaCy, TF-IDF,
scikit-learn, pandas, SQL, Git,
deep learning and PyTorch.

Responsibilities:

Build text classification models.
Perform named entity recognition.
Perform data visualization.
Present findings to the team.
"""


# -----------------------------------------
# 11. FILE READER
# -----------------------------------------

def read_upload(file):

    name = file.name.lower()
    data = file.read()

    if name.endswith(".pdf"):

        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    if name.endswith(".docx"):

        import docx

        document = docx.Document(io.BytesIO(data))

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    return data.decode(
        "utf-8",
        errors="ignore"
    )


def get_text(label, key, sample):

    mode = st.radio(
        f"{label} input",
        ["Paste text", "Upload file"],
        horizontal=True,
        key=key + "_mode"
    )

    if mode == "Upload file":

        file = st.file_uploader(
            f"Upload {label} (PDF, DOCX or TXT)",
            type=["pdf", "docx", "txt"],
            key=key + "_file"
        )

        if file:

            text = read_upload(file)

            if not text.strip():
                st.warning(
                    "No text could be extracted. "
                    "Scanned PDFs require OCR."
                )

            return text

        return ""

    return st.text_area(
        label,
        value=sample,
        height=320,
        key=key + "_text"
    )


# -----------------------------------------
# 12. CLASSIFIER CACHE
# -----------------------------------------

@st.cache_resource
def get_classifier():

    return build_classifier()


# -----------------------------------------
# 13. STREAMLIT INTERFACE
# -----------------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Analyzer")

st.caption(
    "NLP Mini Project | BCA 307T | "
    "Resume and Job Description Matching"
)


with st.sidebar:

    st.header("About Project")

    st.write(
        "This application compares a resume "
        "with a job description using NLP."
    )

    st.markdown("""
    **NLP Concepts Used**

    - Tokenization
    - Stop-word Removal
    - Stemming
    - Bag of Words
    - TF-IDF
    - N-grams
    - Cosine Similarity
    - Information Extraction
    - Naive Bayes
    """)

    st.info(
        "This tool assists resume screening. "
        "It does not replace human evaluation."
    )


left, right = st.columns(2)

with left:

    st.subheader("Resume")

    resume = get_text(
        "Resume",
        "resume",
        SAMPLE_RESUME
    )

with right:

    st.subheader("Job Description")

    jd = get_text(
        "Job Description",
        "jd",
        SAMPLE_JD
    )


# -----------------------------------------
# 14. ANALYZE BUTTON
# -----------------------------------------

if st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
):

    if not resume.strip() or not jd.strip():

        st.error(
            "Please provide both resume "
            "and job description."
        )

        st.stop()

    try:

        result = analyze(resume, jd)

        entities = extract_entities(resume)

        vectorizer, classifier = get_classifier()

        ranked = classify(
            resume,
            vectorizer,
            classifier
        )

        st.divider()

        st.header("Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Overall ATS Score",
            f"{result['score']}/100"
        )

        col2.metric(
            "Skill Coverage",
            f"{result['skill_score'] * 100:.0f}%"
        )

        col3.metric(
            "TF-IDF Similarity",
            f"{result['similarity']:.2f}"
        )

        st.progress(
            min(int(result["score"]), 100)
        )

        verdict_text = get_verdict(result["score"])

        if result["score"] >= 75:
            st.success(verdict_text)

        elif result["score"] >= 50:
            st.warning(verdict_text)

        else:
            st.error(verdict_text)


        # ---------------------------------
        # RESULTS TABS
        # ---------------------------------

        tab1, tab2, tab3, tab4 = st.tabs([
            "Skills Match",
            "Candidate Details",
            "Career Category",
            "NLP Internals"
        ])


        # TAB 1: SKILLS MATCH

        with tab1:

            col1, col2 = st.columns(2)

            col1.markdown("### Matched Skills")

            col1.write(
                ", ".join(result["matched"]) or "None"
            )

            col2.markdown("### Missing Skills")

            col2.write(
                ", ".join(result["missing"]) or "None"
            )

            st.markdown("### All Resume Skills")

            st.write(
                ", ".join(result["resume_skills"]) or "None"
            )

            st.markdown("### Sections Detected")

            st.write(
                ", ".join(result["sections"]) or "None"
            )

            required_sections = [
                "education",
                "experience",
                "projects",
                "skills"
            ]

            missing_sections = [
                section
                for section in required_sections
                if section not in result["sections"]
            ]

            if missing_sections:

                st.warning(
                    "Consider adding sections: "
                    + ", ".join(missing_sections)
                )


        # TAB 2: CANDIDATE DETAILS

        with tab2:

            dataframe = pd.DataFrame(
                [
                    (
                        key,
                        ", ".join(value)
                        if isinstance(value, list)
                        else str(value)
                    )
                    for key, value in entities.items()
                ],
                columns=["Field", "Value"]
            )

            st.table(
                dataframe.set_index("Field")
            )


        # TAB 3: CAREER CATEGORY

        with tab3:

            dataframe = pd.DataFrame(
                ranked,
                columns=["Category", "Probability"]
            )

            st.bar_chart(
                dataframe.set_index("Category")
            )

            st.write(
                f"Predicted Category: "
                f"**{ranked[0][0]}** "
                f"({ranked[0][1] * 100:.1f}%)"
            )

            if st.checkbox(
                "Show classifier evaluation"
            ):

                accuracy, train_size, test_size = (
                    evaluate_classifier()
                )

                st.write(
                    f"Accuracy: {accuracy * 100:.0f}%"
                )

                st.write(
                    f"Training samples: {train_size}"
                )

                st.write(
                    f"Testing samples: {test_size}"
                )

                st.caption(
                    "Small demonstration dataset; "
                    "accuracy is not a reliable estimate "
                    "of real-world performance."
                )


        # TAB 4: NLP INTERNALS

        with tab4:

            st.markdown(
                "### Preprocessing Pipeline"
            )

            st.write(
                "Tokenization → Stop-word Removal → Stemming"
            )

            st.code(
                " ".join(preprocess(resume)[:40]) or "-"
            )

            st.markdown(
                "### Top Word Frequencies"
            )

            frequency_df = pd.DataFrame(
                word_frequency(resume, 10),
                columns=["Word", "Count"]
            )

            if not frequency_df.empty:

                st.bar_chart(
                    frequency_df.set_index("Word")
                )

            st.markdown("### Bag of Words")

            st.write(
                ", ".join(bow_demo(resume))
            )

            st.markdown(
                "### Top Job Description Keywords"
            )

            st.write(
                ", ".join(result["jd_keywords"])
            )

    except Exception as error:

        st.error(
            f"An error occurred: {error}"
  )
                  
