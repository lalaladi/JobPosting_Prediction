import streamlit as st
import re, string
import numpy as np
import nltk
from ibm_watson_machine_learning import APIClient
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# IBM Watson credentials
api_key = 'puKabkitvKXlaGRpojQAXx9GRp9BML00zqKA9RqJtDdU'
location = 'us-south'
wml_credentials = {
    "apikey": api_key,
    "url": f'https://{location}.ml.cloud.ibm.com'
}

# Initialize Watson Machine Learning client
client = APIClient(wml_credentials)
space_id = 'a58633c5-05e9-4686-aa7b-0441a0df945e'
client.set.default_space(space_id)
deployment_id = 'c6306e5c-7605-4598-b2ee-05117b7c61ba'

# Define text preprocessing function
stop_words = set(stopwords.words("english"))
punctuation = list(string.punctuation)
stop_words.update(punctuation)
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r"\n", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www.\S+", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
    text = ' '.join(tokens)
    return text

max_features = 10000
tokenizer = Tokenizer(num_words=max_features)

st.title('Job Posting Prediction')
st.write('Ini merupakan suatu model untuk mengetahui apakah postingan pekerjaan palsu atau tidak')

# Input data
combined_text = st.text_area('Masukkan detail lowongan pekerjaan :')

if st.button('Prediction Result :'):
    processed_text = text_preprocessing(combined_text)
    tokenizer.fit_on_texts([processed_text])
    encoded_docs_testing = tokenizer.texts_to_sequences([processed_text])
    max_sequence_length = 1445  # Sesuaikan dengan panjang yang diharapkan oleh model
    embedded_docs_testing = pad_sequences(encoded_docs_testing, padding='pre', maxlen=max_sequence_length)
    testing_data = np.array(embedded_docs_testing)

    scoring_payload = {"input_data": [{"values": testing_data.tolist()}]}
    predictions = client.deployments.score(deployment_id, scoring_payload)
    scoring_response_json = predictions

    prediction = scoring_response_json['predictions'][0]
    predicted_probabilities = prediction['values'][0][0]
    predicted_class = prediction['values'][0][1]

    if predicted_class == [0]:
        st.success("Not Fraudulent")
    else:
        st.error("Fraudulent")
