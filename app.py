import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set page configuration
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

st.title("💬 Twitter Sentiment Analysis")
st.markdown("Analyze tweet sentiment (Positive or Negative) using Logistic Regression.")

# Sidebar for file upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a sampled Twitter dataset (<300MB)", type=["csv"])

# Column names
column_names = ['target', 'id', 'date', 'query', 'user', 'text']

# Load dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='latin-1', names=column_names)
else:
    st.warning("📂 Please upload a CSV file (sampled under 300MB).")
    st.stop()

# Preprocess text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Convert target: 0 = Negative, 4 = Positive → 0/1
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Show a sample of the data
st.subheader("📋 Sample Data")
st.write(df[['target', 'text', 'clean_text']].sample(5))

# Sentiment distribution
st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots()
sns.countplot(x='target', data=df, ax=ax)
ax.set_xticklabels(['Negative', 'Positive'])
ax.set_ylabel("Tweet Count")
st.pyplot(fig)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['target']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy and Classification Report
st.subheader("✅ Model Performance")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.4f}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("📌 Confusion Matrix")
fig2, ax2 = plt.subplots()
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=ax2)
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)
