# CEFR-LexiProbe: Statistical NLP & Linguistic Complexity Classifier

End-to-end supervised NLP pipeline for English proficiency (CEFR A1–C2) and readability classification. Combines **word/character N-gram TF-IDF vectorization** with **8 psycholinguistic & surface complexity features** (Carroll's Corrected TTR, Coxhead Academic Word List density, syllable morphology, subordination index, Flesch-Kincaid grade) across three supervised classifiers (**Logistic Regression, Linear SVM, Random Forest**).

## Key Features
- **Hybrid Representation (`FeatureUnion`):** Word N-grams (1–2), Character N-grams (3–5 for derivational morphology), and standardized linguistic complexity features.
- **Multi-Model Benchmarking:** Automated stratified train/test evaluation reporting Accuracy, Macro Precision, Macro Recall, Macro F1, and Confusion Matrices.
- **Explainable Token Attribution:** Highlights input words according to linear log-odds coefficients.
- **Hugging Face Datasets Integration:** Pulls from `UniversalCEFR/cefr_sp_en` with a deterministic benchmark fallback.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
