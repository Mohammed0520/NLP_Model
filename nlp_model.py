from flask import Flask, request, jsonify
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
from scipy.special import softmax
import re
import unicodedata
import pandas as pd 
from tqdm.notebook import tqdm
import numpy as np
import tensorflow as tf
app = Flask(__name__)



def pretraiter_dataframe(df):
    # Normaliser le texte des commentaires de clients
    def normaliser_texte(texte):
        # Convertir le texte en minuscules
        texte = texte.lower()
        # Supprimer les accents et les caractères spéciaux
        texte = unicodedata.normalize('NFKD', texte).encode('ASCII', 'ignore').decode('utf-8')
        return texte

    # Supprimer les informations redondantes ou inutiles des commentaires de clients, telles que les liens ou les émoticônes
    def nettoyer_texte(texte):
        # Supprimer les liens
        texte = re.sub(r'http\S+', '', texte)
        # Supprimer les émoticônes
        texte = re.sub(r'[^\w\s]', '', texte)
        return texte

    # Appliquer le prétraitement de données à chaque ligne du dataframe
    df['Text'] = df['Text'].apply(lambda x: nettoyer_texte(normaliser_texte(x)))

    return df

def PFA_model(example):
    # Title: Loading and Using PFA Model
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)

    # Encode the example text
    encoded_text = tokenizer(example, return_tensors='tf')
    output = model(**encoded_text)
    scores = tf.stop_gradient(output[0][0]).numpy()
    scores = softmax(scores)
    sentiment = ['negatif', 'neutre', 'positif'][np.argmax(scores)]

    return sentiment

# Function for obtaining sentiment analysis results for the dataframe
def model_result(df):
    res = {}
    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            text = row['Text']
            myid = row['Id']

            # Perform sentiment analysis using the PFA model
            roberta_result = PFA_model(text)
            res[myid] = {'Id': myid, 'Text': text, 'ProductId': row['ProductId'], 'Score': roberta_result}
        except RuntimeError:
            print(f'Broke for id {myid}')

    # Convert the results into a dataframe
    results_df = pd.DataFrame(res).T

    return results_df[['Id', 'Text', 'ProductId', 'Score']]

@app.route('/nlp_model', methods=['POST'])
def nlp_model():
    url_str = request.form['url_str']
    data = pd.read_csv(url_str)
    data = data.head(10)
    data_prep = pretraiter_dataframe(data)
    results = model_result(data_prep)
    return jsonify(results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
