
import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Restaurant Review Sentiment Analysis", page_icon="🍽️")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("count_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

st.title("🍽️ Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review to predict its sentiment.")

review = st.text_area("Enter Your Review", height=120)

if st.button("Predict Sentiment"):
    if review.strip():
        cleaned_review = clean_text(review)
        review_vector = vectorizer.transform([cleaned_review])
        prediction = model.predict(review_vector)[0]

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")
    else:
        st.warning("Please enter a review.")

st.caption("Powered by Bernoulli Naive Bayes")
