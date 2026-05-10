![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![NLP](https://img.shields.io/badge/NLP-sentence--transformers-orange) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

# Supply Chain Customer Satisfaction — NLP Classification

NLP + tabular classification pipeline predicting customer satisfaction from supply chain data and free-text reviews.

---

## Task

**Text + Tabular Classification**

---

## Architecture

```
Free-text Reviews + Structured Features → SBERT Embeddings + Feature Eng → Late Fusion → LightGBM
```

---

## Key Features

- Hybrid NLP + tabular feature fusion
- Sentence-BERT embeddings for free-text reviews
- LightGBM on concatenated feature vectors
- Late-fusion ensemble (text branch + structured branch)
- SHAP explanation of top satisfaction drivers

---

## Dataset

Supply chain CSAT dataset (Kaggle)

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # Baseline model
│   └── model_advanced.py      # Advanced model
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup
├── reports/
│   └── references.md          # Verified references
├── deliverables/
│   └── presentation.html      # Self-contained HTML
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/supply-chain-satisfaction-nlp.git
cd supply-chain-satisfaction-nlp
pip install -r requirements.txt

# See data/README.md for dataset download
python src/model_baseline.py
python src/model_advanced.py
```

---

## Tech Stack

`scikit-learn · sentence-transformers · LightGBM · pandas`

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
