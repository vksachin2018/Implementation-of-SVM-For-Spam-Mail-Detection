# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the spam email dataset and preprocess the labels into numerical values.

2. Convert the email text messages into numerical feature vectors using TF-IDF.

4. Split the dataset into training and testing sets and train the SVM classifier.

5. Use the trained model to predict spam or non-spam emails and evaluate the accuracy.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Gokul sachin k
RegisterNumber:  212223220025
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("spam.csv", encoding="latin-1")

df = df[['v1','v2']]
df.columns = ['label','message']

df['label'] = df['label'].map({'ham':0, 'spam':1})

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_vec, y_train)

y_pred = svm_model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

email = ["Congratulations! You have won a free lottery ticket"]

email_vec = vectorizer.transform(email)
prediction = svm_model.predict(email_vec)

if prediction[0] == 1:
    print("Prediction: Spam Mail")
else:
    print("Prediction: Not Spam")
```

## Output:

<img width="1365" height="268" alt="Screenshot 2026-03-09 154440" src="https://github.com/user-attachments/assets/46fb1cb1-e777-47dc-af1e-8f97c6090f70" />

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
