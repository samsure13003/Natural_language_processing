# 🍽️ Restaurant Review Sentiment Classification — NLP with Bag of Words

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green?style=flat)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat)
![Models](https://img.shields.io/badge/Models-7-purple?style=flat)

## 📌 Project Overview

This project performs **sentiment analysis** on restaurant reviews using **Natural Language Processing (NLP)**. The goal is to classify each review as either **positive (1)** or **negative (0)** based on the text content.

The text data is converted into numerical features using the **Bag of Words (BoW)** model, and **7 different classification algorithms** are trained and evaluated to find the best-performing model.

---

## 📂 Dataset

- **File:** `Restaurant_Reviews.tsv`
- **Total Reviews:** 1,000
- **Columns:**
  - `Review` — Raw text of the restaurant review
  - `Liked` — Target label (`1` = Positive, `0` = Negative)
- **Class Distribution:** 500 Positive / 500 Negative (perfectly balanced)

---

## 🛠️ Tech Stack

| Category | Tools / Libraries |
|---|---|
| Language | Python 3.x |
| Data Handling | NumPy, Pandas |
| NLP | NLTK (Stopwords, PorterStemmer), Regex (`re`) |
| Feature Extraction | Scikit-learn `CountVectorizer` |
| Machine Learning | Scikit-learn classifiers |
| Evaluation | Accuracy Score, Confusion Matrix |

---

## ⚙️ Workflow

### 1. Text Cleaning & Preprocessing

Each review goes through the following steps:
- Remove all non-alphabetic characters using `re.sub`
- Convert to **lowercase**
- **Tokenize** (split into individual words)
- Remove **stopwords** (except `"not"` — crucial for sentiment)
- Apply **Porter Stemming** to reduce words to their root form

### 2. Bag of Words (BoW) — Feature Extraction

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
```

- The cleaned corpus is converted into a **sparse matrix** of word frequencies
- Top **1500 most frequent words** are kept as features
- Each review becomes a vector of 1500 numbers

### 3. Train-Test Split

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

- **80% Training** / **20% Testing** split
- 800 reviews for training, 200 for testing

### 4. Model Training & Evaluation

Seven classifiers are trained and evaluated using **Accuracy Score** and **Confusion Matrix**.

---

## 📊 Model Results

| Model | Accuracy | TP | TN | FP | FN |
|---|---|---|---|---|---|
| **SVM (Linear Kernel)** | **78.5%** | 79 | 78 | 19 | 24 |
| Logistic Regression | 77.5% | 75 | 80 | 17 | 28 |
| Kernel SVM | 77.0% | 66 | 88 | 9 | 37 |
| Decision Tree | 76.0% | 71 | 81 | 16 | 32 |
| Naive Bayes | 73.0% | 91 | 55 | 42 | 12 |
| Random Forest | 73.0% | 62 | 84 | 13 | 41 |
| KNN | 66.0% | 47 | 85 | 12 | 56 |

> **TP** = True Positive | **TN** = True Negative | **FP** = False Positive | **FN** = False Negative

### 🏆 Best Model: SVM (Linear Kernel) — 78.5% Accuracy

---

## 🔍 Confusion Matrix Summary

### SVM (Linear Kernel) — Best Performer
```
Predicted →     Negative  Positive
Actual Negative    78        19
Actual Positive    24        79
```

### Naive Bayes — Highest Recall for Positives
```
Predicted →     Negative  Positive
Actual Negative    55        42
Actual Positive    12        91
```
> Naive Bayes catches the most positive reviews (high recall) but at the cost of more false positives.

---

## 📁 Project Structure

```
📦 nlp-restaurant-sentiment/
├── 📄 Restaurant_Reviews.tsv                          # Dataset
├── 📓 natural_language_processing_logistic_regression.ipynb
├── 📓 natural_language_processing_naive_bays.ipynb
├── 📓 natural_language_processing_Decision_tree.ipynb
├── 📓 natural_language_processing_KNN.ipynb
├── 📓 natural_language_processing_kernel_svm.ipynb
├── 📓 natural_language_processing_support_vector.ipynb
├── 📓 natural_language_processing_random_forest.ipynb
└── 📄 README.md
```

---

## 🚀 How to Run

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/nlp-restaurant-sentiment.git
cd nlp-restaurant-sentiment
```

2. **Install dependencies:**
```bash
pip install numpy pandas matplotlib scikit-learn nltk
```

3. **Download NLTK stopwords** (runs automatically in notebooks):
```python
import nltk
nltk.download('stopwords')
```

4. **Open any notebook** in Jupyter and run all cells:
```bash
jupyter notebook natural_language_processing_logistic_regression.ipynb
```

---

## 💡 Key Takeaways

- The **Bag of Words** model, despite being simple, provides a strong baseline for sentiment analysis.
- Keeping `"not"` in the stopwords list is critical — removing it would flip the sentiment of phrases like *"not good"*.
- **SVM with a linear kernel** performed best on this dataset, making it the recommended model.
- **Naive Bayes** had the highest recall for positive reviews, which is useful if missing positive sentiment is more costly.
- **KNN** performed worst, likely because high-dimensional BoW vectors suffer from the curse of dimensionality.

---

## 🔮 Future Improvements

- Try **TF-IDF** instead of Bag of Words for better feature weighting
- Use **Word Embeddings** (Word2Vec, GloVe) for richer text representation
- Apply **deep learning models** (LSTM, BERT) for sequence-aware sentiment analysis
- Perform **hyperparameter tuning** (GridSearchCV) on top models

---

## 👤 Author

**Samsur**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
