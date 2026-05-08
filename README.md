# CAD — Cognitive Authenticity Detection

> An AI system that evaluates the **depth of student understanding**, not just answer correctness.

---

## Overview

CAD analyses short student answers and outputs a **Genuine Understanding Score (GUS)** — a 0–100 metric that reflects how deeply a student understands a concept, rather than whether they gave the "right" words.

**Pipeline:**
```
Student Answer → Preprocessing → Feature Extraction → ML Classifier → Bayesian Calibration → GUS Score
```

**Three model tiers:**
| Model | Description | Speed | Accuracy |
|-------|-------------|-------|----------|
| Rule-based | Word count + causal language heuristics | Instant | Baseline |
| TF-IDF + LR | TF-IDF vectors + linguistic features + Logistic Regression | Fast | Good |
| DistilBERT | Fine-tuned transformer, best semantic understanding | Moderate | Best |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For CPU-only PyTorch (smaller install):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Train the TF-IDF model

```bash
python src/train.py --model tfidf
```

This saves `results/tfidf_model.pkl` and prints evaluation metrics.

### 3. Run the web UI

```bash
python ui/app.py
```

Open **http://localhost:5000** in your browser.

### 4. Command-line inference

```bash
# Single answer
python src/infer.py --answer "Photosynthesis converts light to glucose via the Calvin cycle in chloroplasts."

# With question context
python src/infer.py --answer "F=ma means force equals mass times acceleration." --question "Explain Newton's second law."

# Batch inference on CSV
python src/infer.py --csv data/answers.csv --output results/batch_predictions.csv

# Use BERT model (requires training first)
python src/infer.py --model bert --answer "Your answer here"
```

---

## Training

### TF-IDF + Logistic Regression (recommended for demo)
```bash
python src/train.py --model tfidf --data data/answers.csv
```

### Fine-tune DistilBERT (GPU recommended)
```bash
python src/train.py --model bert --epochs 3 --data data/answers.csv
```

### Train everything
```bash
python src/train.py --model all
```

Results (metrics JSON, model files, comparison chart) are saved to `results/`.

---

## Project Structure

```
CAD_project/
├── data/
│   └── answers.csv          # Labelled Q&A dataset (Deep / Shallow)
├── src/
│   ├── __init__.py
│   ├── utils.py             # Data loading, preprocessing, GUS computation
│   ├── model.py             # RuleBasedCAD, TFIDFModel, BERTModel, BayesianConfidenceWrapper
│   ├── train.py             # Training script (all models)
│   └── infer.py             # Inference engine + CLI
├── ui/
│   ├── app.py               # Flask web application
│   ├── templates/
│   │   └── index.html       # Main UI template
│   └── static/
│       ├── css/style.css    # Stylesheet
│       └── js/app.js        # Frontend logic
├── notebooks/
│   └── demo.ipynb           # Jupyter demo notebook
├── results/                 # Auto-created: saved models, metrics, plots
├── .vscode/
│   ├── launch.json          # Debug configurations
│   └── tasks.json           # Build/run tasks
├── requirements.txt
├── Dockerfile
├── .env.example
└── README.md
```

---

## GUS Score Interpretation

| GUS Range | Verdict | Meaning |
|-----------|---------|---------|
| 70 – 100 | Deep Understanding | Causal reasoning, conceptual links, specific detail |
| 40 – 69 | Moderate Understanding | Partial depth, some correct ideas, lacks full reasoning |
| 0 – 39 | Shallow Understanding | Superficial, definitional only, likely memorised |

**GUS formula:**
```
GUS = calibrated_prob × 100
    + causal_language_boost (up to +10)
    + connector_language_boost (up to +6)
    + formula/number_boost (+4)
    + lexical_richness_boost (up to +8)
    - short_answer_penalty (-15 if < 10 words)
```

---

## Docker

```bash
# Build (trains TF-IDF model inside container)
docker build -t cad-app .

# Run
docker run -p 5000:5000 cad-app
```

---

## Evaluation Metrics

After training, results are saved to `results/`:
- `tfidf_metrics.json` — accuracy, precision, recall, F1, ROC-AUC, 5-fold CV F1
- `tfidf_predictions.json` — per-sample predictions on test set
- `model_comparison.png` — bar chart comparing all trained models
- `gus_distribution.png` — GUS score distribution (from notebook)

Target: **F1 ≥ 0.80** on the labelled test set.

---

## Adding More Data

Edit `data/answers.csv` with additional rows following the schema:

```
Question,Answer,Label
"Your question here","Student's answer here","Deep"
"Your question here","Short answer","Shallow"
```

Labels: `Deep` = genuine conceptual understanding | `Shallow` = surface-level or memorised

---

## Team

**Team CerebraX** — CAD prototype v1.0  
Powered by scikit-learn · HuggingFace Transformers · Flask
