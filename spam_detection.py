print("=" * 60)
print("SPAM MESSAGE DETECTION USING MACHINE LEARNING")
print("=" * 60)

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------- DATA LOAD ----------------
df = pd.read_csv("dataset/spam.csv", encoding="latin-1")
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(df.head())

X = df['message']
y = df['label']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- VECTORIZATION ----------------
vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ---------------- MODEL 1: NAIVE BAYES ----------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

print("\n===== NAIVE BAYES RESULTS =====")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print("Precision:", precision_score(y_test, nb_pred))
print("Recall:", recall_score(y_test, nb_pred))
print("F1 Score:", f1_score(y_test, nb_pred))

nb_prob = nb_model.predict_proba(X_test_vec)[:, 1]
print("ROC-AUC:", roc_auc_score(y_test, nb_prob))

print("Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))

# Cross validation
cv_scores = cross_val_score(nb_model, X_train_vec, y_train, cv=5)
print("Cross Validation Mean:", cv_scores.mean())

# ---------------- MODEL 2: LOGISTIC REGRESSION ----------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)

print("\n===== LOGISTIC REGRESSION RESULTS =====")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("F1 Score:", f1_score(y_test, lr_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# ---------------- USER INPUT PREDICTION ----------------
msg = input("\nEnter a message: ")

msg_vec = vectorizer.transform([msg])

result = nb_model.predict(msg_vec)

if result[0] == 1:
    print("Prediction: SPAM 🚨")
else:
    print("Prediction: NOT SPAM ✅")

print("\nProject Completed Successfully!")
print("Developed by: Dhanusri Ramesh")