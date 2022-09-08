from ctypes import alignment
from tkinter import CENTER
from turtle import width
import streamlit as st
import requests as r
from os.path import dirname, join, realpath
import joblib
from langdetect import detect

# add banner image
st.header("Gender-Based Violence Tweet App")
st.image("Image/gbv.png", width = 400)
st.write('A machine learning App used to predict whether tweets are among the category or type, which are sexual violence, emotional violence, harmful traditional practices, physical violence and economic violence. ')

# form to collect news content
my_form = st.form(key="tweet_form")
tweet = my_form.text_input("Input your tweet here")
submit = my_form.form_submit_button(label="make prediction")


# load the model and count_vectorizer

with open(
    join(dirname(realpath(__file__)), "Model/KNNClassifier.pkl"), "rb"
) as m:
    model = joblib.load(m)

with open(join(dirname(realpath(__file__)), "Preprocessing/vectorizer_tfidf.pkl"), "rb") as f:
    vectorizer = joblib.load(f)

classifiers = {0: "Harmful_Traditional_practice", 1: "Physical_violence", 2: "economic_violence", 3:"emotional_violence", 4:"sexual_violence"}

#0 - 'Harmful_Traditional_practice', 1 - 'Physical_violence', 2 - 'economic_violence', 3 - 'emotional_violence', 4 - 'sexual_violence'
if submit:

    if detect(tweet) == "en":

        # transform the input
        transformed_tweet = vectorizer.transform([tweet])

        # perform prediction
        prediction = model.predict(transformed_tweet)
        output = int(prediction[0])

        probas = model.predict_proba(transformed_tweet)
        probability = "{:.2f}".format(float(probas[:, output]))

   

        # Display results of the NLP task
        st.header("Results")
        
        if output == 1:
            st.write("The Gender-based Violence of the tweet is {} ".format(classifiers[output]))
            st.write("Probability of the tweet is {}   ".format(probability))
        elif output == 2:
            st.write("The Gender-based Violence of the tweet is {}  ".format(classifiers[output]))
            st.write("Probability of the tweet is {}   ".format(probability))
        elif output == 3:
            st.write("The Gender-based Violence of the tweet is {}  ".format(classifiers[output]))
            st.write("Probability of the tweet is {}   ".format(probability))
        elif output == 4:
            st.write("The Gender-based Violence of the tweet is {} ".format(classifiers[output]))
            st.write("Probability of the tweet is {}   ".format(probability))
        elif output == 0:
            st.write("The Gender-based Violence of the tweet is {}  ".format(classifiers[output]))
            st.write("Probability of the tweet is {}   ".format(probability))
        else:
            st.write(" ⚠️ The tweet is not in categories of Gender based violence. Please make sure the input is based on these categories")

    else:
        st.write(" ⚠️ The tweet is not in English. Please make sure the input is based on english language")

url = "https://drive.google.com/drive/folders/10yILBCt0_lw1IFZ5LiHU7cWDC-q659ct"
st.write("Developed with ❤️ by [GROUP 6](%s)" % url)
