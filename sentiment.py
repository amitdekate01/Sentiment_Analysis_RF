import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


df = pd.read_csv('IMDB Dataset.csv')
df = df.sample(25000, random_state=1).reset_index(drop=True)

def clean_text(text):
    text = re.sub('<.*?>', '', text)
    text = re.sub(r"https?\S+|www.\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df['review'] = df['review'].apply(clean_text)

#removing stop words
stop_words = set(stopwords.words('english'))
df['review'] = df['review'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

#Encoding labels
le = LabelEncoder()
df['sentiment'] = le.fit_transform(df['sentiment'])

#train test split
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=1)

#Tf-idf vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_vec, y_train)

y_pred = rf.predict(X_test_vec)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification report: ", classification_report(y_test, y_pred, target_names=['negative', 'positive']))

