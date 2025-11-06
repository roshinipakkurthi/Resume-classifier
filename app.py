import os
import numpy as np
import pandas as pd
import pdfplumber
import re
import nltk
import pytesseract
import joblib
from pdf2image import convert_from_path
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Dataset
df = pd.read_csv(r"C:\Users\HP\Downloads\resume_dataset_expanded.csv", dtype=str, low_memory=False)
df = df.rename(columns={"Resume Text": "text", "Job Title": "title"})
df = df[["text", "title"]].dropna().drop_duplicates()

# Extract job-related words
job_titles = set(df["title"].str.lower().str.split().explode().unique())
custom_stop_words = set(stopwords.words("english")) - job_titles
lemmatizer = WordNetLemmatizer()

# Resume-Specific Keywords
resume_keywords = {"experience", "education", "skills", "projects", "certifications", "summary"}

def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9+\-/ ]", "", text).lower()
    words = word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(word) for word in words if word not in custom_stop_words and len(word) > 2])

df["cleaned_text"] = df["text"].apply(clean_text)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=15000, min_df=2, max_df=0.9)
X = vectorizer.fit_transform(df["cleaned_text"])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["title"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = SVC(kernel="linear", probability=True, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "resume_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

model = joblib.load("resume_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    if not text.strip():
        images = convert_from_path(file_path)
        text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return text

def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()
    except:
        return ""

def is_resume(text):
    words = set(word_tokenize(text.lower()))
    matched_keywords = words.intersection(resume_keywords)
    return len(matched_keywords) >= 2

def predict_job_role(text):
    if not is_resume(text):
        return "Not a Resume"
    cleaned_text = clean_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    probabilities = model.predict_proba(vectorized_text)[0]
    top_prediction_idx = np.argmax(probabilities)
    top_prediction = label_encoder.inverse_transform([top_prediction_idx])[0]
    confidence = probabilities[top_prediction_idx]
    return top_prediction if confidence > 0.3 else "Not a Resume"

def summarize_resume(text, num_sentences=3):
    sentences = sent_tokenize(text)
    important_sentences = sorted(sentences[:num_sentences], key=len)  # Choose the shortest meaningful sentences
    return " ".join(important_sentences)

@app.route("/")
def index():
    return render_template("resume.html")  # ✅ Serves the frontend page

@app.route("/upload", methods=["POST"])
def upload():
    if "resume" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["resume"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file.filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        text = extract_text_from_pdf(file_path) if filename.endswith(".pdf") else extract_text_from_txt(file_path)
        if not text.strip():
            return jsonify({"error": "No readable text found in the file"}), 400

        job_role = predict_job_role(text)
        summary = summarize_resume(text)

        return jsonify({"job_role": job_role, "summary": summary})

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
