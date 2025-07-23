import re
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

with st.spinner('Wait for it...'):
    time.sleep(3)  # Simulating a process delay

st.balloons()  # Celebration balloons

st.sidebar.title("𝓟𝓱𝓲𝓼𝓱🐠𝓲𝓷𝓰  𝓓𝓮𝓽𝓮𝓬𝓽🔍𝓻")
select=st.sidebar.selectbox('Checking Type  :-', ["Text","URL"])

if select == "URL":
    
    st.header("URL Detector")

    clf = joblib.load('url_botnet_detector1.pkl')

    df = pd.read_csv('malicious_phish.csv')
    df.info()
    df["label"].dropna()

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['url'])

    df['label'] = df['label'].apply(lambda x: 1 if x == "phishing" or x =="defacement" else 0)
    y=df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    new_url=st.text_input("Enter URL","")

    new_url_vectorized = vectorizer.transform([new_url])
    print(clf.predict(new_url_vectorized)[0])
    #Predict
    if clf.predict(new_url_vectorized)[0]==1:
        image_url = "https://t3.ftcdn.net/jpg/00/99/41/66/360_F_99416675_nLarXVhXt5gozLTlndmEahTsjUDp8QGm.jpg"
        st.image(image_url, use_column_width=True)
    else:
        image_url = "https://th.bing.com/th/id/OIP.KUjjmo3INScU1BBfLyfcLwHaHX?w=213&h=212&c=7&r=0&o=5&dpr=1.5&pid=1.7"
        st.image(image_url, use_column_width=True)        

elif select == "Text":
    
    st.header("Text Detector")

    model=load_model("Phishing.keras")

    data = pd.read_csv('email.csv')  

    data['label'] = data['label'].replace(
        to_replace=['ham', 'spam'], 
        value=[0, 1])

    # Preprocessing function
    def preprocess_text(text):
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove non-alphabetic characters
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        # Lowercase and strip
        text = text.lower().strip()
        return text

    # Apply preprocessing
    data['text'] = data['text'].apply(preprocess_text)

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    msg=st.text_input("Enter Text","")

    processed_Msg = preprocess_text(msg)

    vecterized_text = vectorizer.transform([processed_Msg])

    f32msg=vecterized_text.toarray().astype('float32')

    prediction = model.predict(f32msg)
    
    if prediction[0] > 0.5:
        image_url = "https://t3.ftcdn.net/jpg/00/99/41/66/360_F_99416675_nLarXVhXt5gozLTlndmEahTsjUDp8QGm.jpg"
        st.image(image_url, use_column_width=True)
    else:
        image_url = "https://th.bing.com/th/id/OIP.KUjjmo3INScU1BBfLyfcLwHaHX?w=213&h=212&c=7&r=0&o=5&dpr=1.5&pid=1.7"
        st.image(image_url, use_column_width=True)
else :
    pass