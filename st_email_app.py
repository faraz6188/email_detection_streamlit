import joblib

model=joblib.load('ed.h5')
vec = joblib.load('vectorizer.h5')

def predict_spam(text):
    text_vector = vec.transform([text])
    prediction = model.predict(text_vector)
    return prediction[0]

import streamlit as st


st.title("Spam Detection System by Faraz")

st.sidebar.title("Login here")
st.sidebar.text_input("Mail")
st.sidebar.text_input("Pass")
st.sidebar.button("submit")

user_input = st.text_area("Enter text to classify as Spam or Ham")

if st.button("Predict"):
    result = predict_spam(user_input)
    if result == 1:
        st.error("This text is classified as Spam.")
    else:
        st.success("This text is classified as Ham.")

st.header("Rate my website")
st.select_slider("Rating",["Worst","Bad", "Good","Excellent"])
if st.button("Submit"):
    st.success("Submitted successfully!")
    st.balloons()
