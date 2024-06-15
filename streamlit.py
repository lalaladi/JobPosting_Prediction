import streamlit as st
import re, string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from ibm_watson_machine_learning import APIClient

# Download NLTK data
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# IBM Watson credentials
api_key = 'vD5MYd1qlPmuSio9ubmytdXmbUApziO74nS969CDjtAo'
location = 'us-south'
wml_credentials = {
    "apikey": api_key,
    "url": f'https://{location}.ml.cloud.ibm.com'
}

# Initialize Watson Machine Learning client
client = APIClient(wml_credentials)
space_id = '9b602fc8-d468-41cb-a19a-b140f78e4d00'
client.set.default_space(space_id)
deployment_id = '91213a22-e99b-4644-bf07-9501dbf54991'

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
    text = re.sub(r"\\n", " ", text)
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

st.title('Text Classification with IBM Watson')
user_input = st.text_area("Enter text to classify")

if st.button('Classify'):
    if user_input:
        processed_text = text_preprocessing(user_input)
        tokenizer.fit_on_texts([processed_text])
        encoded_docs_testing = tokenizer.texts_to_sequences([processed_text])
        embedded_docs_testing = pad_sequences(encoded_docs_testing, padding='pre', maxlen=100000)
        testing_data = np.array(embedded_docs_testing)

        scoring_payload = {"input_data": [{"values": testing_data.tolist()}]}
        predictions = client.deployments.score(deployment_id, scoring_payload)
        scoring_response_json = predictions

        prediction = scoring_response_json['predictions'][0]
        predicted_probabilities = prediction['values'][0][0]
        predicted_class = prediction['values'][0][1]

        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Predicted probabilities: {predicted_probabilities}")
    else:
        st.write("Please enter text to classify.")
