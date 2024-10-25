import streamlit as st
import numpy as np
import pickle
import gensim
from sklearn.linear_model import LogisticRegression

def convert_text_to_vector(text,model):
    words=text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

def emoji_animation(emoji):
    return f"""
    <style>
        .emoji {{
            position: absolute;
            font-size: 50px;
            animation: wave 5s infinite;
        }}

        @keyframes wave {{
            0% {{ transform: translateY(0); }}
            50% {{ transform: translateY(-20px); }}
            100% {{ transform: translateY(0); }}
        }}
    </style>
    <div class="emoji" style="left: {np.random.randint(0, 90)}%; top: {np.random.randint(0, 90)}%;">{emoji}</div>
    """

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",  # You can replace this with a path to an image file if needed
    layout="centered"    # You can also use "centered" for a more compact layout
)

st.title("Emotion Classification APP")

st.info("""This app takes your text input and classifies text emotion into three categories (Joy, Fear and Anger) as it is trained on 
        these 3 classes only. """)

input_text=st.text_input("Please Enter your sentence")
try:
    if input_text:
        with open("weights/emotion_classifier_weights_word_2_vec.model","rb") as file:
            word_2_vec_model=pickle.load(file)

        vectors=convert_text_to_vector(input_text,word_2_vec_model)
        with open(r"weights/linear_regression_model.pkl","rb") as lr_model_file:
            lr_model=pickle.load(lr_model_file)
            predicted_emotion=lr_model.predict([vectors])
            
            if predicted_emotion[0]==0:
                st.markdown("**Emotion: Anger** 😡")
                st.markdown(emoji_animation("😡"), unsafe_allow_html=True)
            elif predicted_emotion[0]==1:
                st.markdown("**Emotion: Joy** 😄")
                st.markdown(emoji_animation("😄"), unsafe_allow_html=True)
            elif predicted_emotion[0]==2:
                st.markdown("**Emotion: Fear** 😨")
                st.markdown(emoji_animation("😨"), unsafe_allow_html=True)

except Exception as e:
    print("Exception -",e)
    st.info("Sorry we run into a problem")