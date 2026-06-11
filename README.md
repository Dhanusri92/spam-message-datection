# 📩 Smart Spam Message Detector (Machine Learning Project)

## 🚀 Overview
This project is a machine learning-based system that automatically classifies SMS/text messages as **Spam 🚨** or **Not Spam (Ham) ✅**.

The goal is to demonstrate how Natural Language Processing (NLP) techniques can be used to solve real-world text classification problems.

---

## 🧠 Problem Statement
With increasing digital communication, spam messages have become a major issue.  
This project aims to build a smart system that can accurately detect spam messages using machine learning algorithms.

---

## ⚙️ Technologies Used
- Python 🐍  
- Pandas  
- Scikit-learn  
- NLP (CountVectorizer)

---

## 🤖 Machine Learning Models
- Naive Bayes Classifier (Primary Model)
- Logistic Regression (Comparison Model)

---

## 📊 Workflow

1. Data Collection (SMS Spam Dataset)
2. Data Cleaning & Preprocessing
3. Label Encoding (Spam = 1, Ham = 0)
4. Text Vectorization using CountVectorizer
5. Train-Test Split
6. Model Training
7. Model Evaluation
8. Performance Comparison
9. Real-time Message Prediction

---

## 📈 Evaluation Metrics Used
To ensure model quality, the following metrics were used:

- Accuracy Score
- Precision Score
- Recall Score
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- Cross Validation Score

---

## 💡 Key Features
- Compares two ML models for better understanding
- Real-time message prediction
- Complete NLP pipeline
- Beginner-friendly but industry-relevant project structure

---

## 🧪 Sample Prediction

Input:
> "Congratulations! You won a free prize"

Output:
> 🚨 SPAM

Input:
> "Are you coming home today?"

Output:
> ✅ NOT SPAM

---

## 🚀 How to Run

```bash
python spam_detection.py