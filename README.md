# Resume-classifier
A machine learning web application to automatically classify resumes into job categories using NLP & scikit-learn.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation & Setup](#installation--setup)

## Overview
This project takes uploaded resume documents (e.g., PDF, DOCX) and predicts a job category/class based on the content. It uses natural language processing (tokenization, vectorisation) together with a trained classification model. The app is built with a simple web UI for easy uploading and result display.

## Features
- Upload a resume file and get back a predicted job category.  
- Pre-processing pipeline: text extraction → cleaning → TF-IDF vectorisation.  
- Pre-trained model stored as `model.pkl` for fast inference.  
- Web front end (Flask/Streamlit or similar) for user interaction.  
- Easily extendable to new categories or improved models.

## Architecture
1. **Text Extraction & Pre-processing** – Extract text from uploaded file → clean (remove punctuation, stopwords) → convert to lower case → lemmatize (if used).  
2. **Vectorisation** – Use the saved vectorizer (e.g., `tfidf_vectorizer.pkl` or `vectorizer.pkl`) to transform the cleaned text into features.  
3. **Classification** – Use the saved model (`model.pkl`) plus label encoder (`label_encoder.pkl`) to predict the job category.  
4. **Web Interface** – Upload interface, display prediction and probability.  
5. **Static / Templates** – HTML/CSS for UI (`/templates`, `/static` folders) and upload logic (`/uploads` folder for temporary storage).  
6. **NLP Data** – `nltk_data/corpora` used for stop-words, etc.

## Installation & Setup
### Prerequisites
- Python 3.7+  
- Virtual environment (recommended)

### Steps
```bash
# Clone the repository
git clone https://github.com/roshinipakkurthi/Resume-classifier.git
cd Resume-classifier

# Create & activate virtual env
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK corpora (if not already present)
python -m nltk.downloader stopwords wordnet

# Run the application
python app.py

Resume-classifier/
│
├─ nltk_data/            ← NLTK corpora used
│  └─ corpora/
│
├─ resume/               ← sample resumes / dataset (optional)
│
├─ static/               ← CSS, JS, images for UI
│
├─ templates/            ← HTML templates for web app
│
├─ uploads/              ← directory to store uploaded files
│
├─ .gitattributes
├─ .gitignore
├─ README.md             ← this file
├─ app.py                ← main application entry point
├─ label_encoder.pkl     ← saved label encoder
├─ model.pkl             ← saved classification model
├─ resume_classifier.pkl ← saved classifier (if separate)
├─ tfidf_vectorizer.pkl  ← saved vectoriser
└─ requirements.txt      ← list of Python dependencies
