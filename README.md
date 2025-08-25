# AI‑Powered Screenshot Classifier 

Automates classification of screenshots for investigations. Identifies **chats, transactions, threats, adult content**, and more using OCR and ML.

## Features

* Multi‑language OCR
* Entity recognition (names, emails, accounts)
* Multilabel classification
* REST API + Streamlit UI

## Tech Stack

* OCR: Tesseract / EasyOCR
* NLP: Hugging Face Transformers, spaCy
* ML: PyTorch
* API/UI: FastAPI + Streamlit
* DB: SQLite/PostgreSQL

## Quick Start

```bash
# Setup
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run API
uvicorn app.api:app --reload

# Run UI
treamlit run ui/dashboard.py
```

## Project Structure

```
AI_SS_Classifier/
├─ app/        # API logic
├─ ui/         # Dashboard
├─ models/     # Saved models
├─ pipelines/  # Train/evaluate
├─ utils/      # OCR & NLP helpers
└─ data/       # Screenshots/data
```

##Model Train Confusion Matrx
<img width="934" height="794" alt="image" src="https://github.com/user-attachments/assets/a6af9004-eb84-456b-8302-cfd63677dcaa" />

