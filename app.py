import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pickle_in = open('emotion-classifier.pkl', 'rb')
model = pickle.load(pickle_in)

emotions_emoji_dict = {"anger":"😠","fear":"😨😱", "joy":"😄", "love": "🥰😍", "sadness":"😔","surprise":"😮"}



def predict_emotion(docs):
    results = model.predict([docs])
    return results[0]

def get_probability(docs):
    results = model.predict_proba([docs])
    return results



def main():
    st.title('Emotion Classifier App')
    menu = ['Home', 'Monitor', 'About']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Home-Emotion In Text')

        with st.form(key = 'emotion_clf form'):
            raw_text = st.text_area('Type Here')
            submit_text = st.form_submit_button(label='Submit')
        
        if submit_text:
            col1, col2 = st.columns(2)

            prediction = predict_emotion(raw_text)
            probability = get_probability(raw_text)


            with col1:
                st.success('Original Text')
                st.write(raw_text)


                st.success('Prediction')
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f'{prediction.capitalize()}  {emoji_icon} ')

            with col2:
                st.success('Prediction Probability')
                # st.write(probability)
                proba_df = pd.DataFrame(probability, columns=model.classes_)
                # st.write(proba_df.T)
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ['emotions', 'probability']


                fig, ax = plt.subplots()
                colors = ['IndianRed', 'Teal', 'olive', 'Fuchsia', 'purple', 'brown']
                ax.bar(x=proba_df_clean['emotions'], height=proba_df_clean['probability'], color=colors)

                st.pyplot(fig)


    if choice == 'Monitor':
        st.subheader('Monitor App')






if __name__ == '__main__':
    main()
