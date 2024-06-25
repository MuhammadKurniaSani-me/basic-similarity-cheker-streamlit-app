# import all libraries
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import BernoulliRBM
import pandas as pd
import string
import re
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.tokenize import RegexpTokenizer

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import ssl

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# normalization
MinMaxscaler = MinMaxScaler()
MABSscaler = MaxAbsScaler()

# preprocessing


# Stemming, lemmatisation, and stopword removal
factory = StemmerFactory()
sastrawi_stemmer = factory.create_stemmer()

nltk_lemmatizer = WordNetLemmatizer()

stopwords = nltk.corpus.stopwords.words('indonesian')


# preprocessing functions
@st.cache_data
def remove_digits(word):
    pattern = '[0-9]'
    return re.sub(pattern, '', word)


@st.cache_data
def remove_punctuation(word_token):
    return word_token if word_token not in string.punctuation else ""


@st.cache_data
def tokenize_and_lowercase(doc):
    return [word.lower() for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]


@st.cache_data
def stemmer(word_token):
    return sastrawi_stemmer.stem(word_token)


@st.cache_data
def lemmatizer(stemmed_word):
    return nltk_lemmatizer.lemmatize(stemmed_word)


@st.cache_data
def finishing(tokens_result):
    return list(filter(lambda token: token.isalpha() or token.isdigit() or token.isspace(), tokens_result))


@st.cache_data
def stringify(token_data):
    return ' '.join(token_data)


@st.cache_data
def preprocessing(corpus_data):
    str_data = []
    token_data = []

    for idx, doc in enumerate(corpus_data):

        tokens_result = []

        # process 1 & 2
        lowercased_tokens = tokenize_and_lowercase(doc)

        for word_token in lowercased_tokens:

            # process 3
            if word_token not in stopwords:
                # process 4 & 5
                stem = stemmer(remove_punctuation(remove_digits(word_token)))

                # process 6
                lem = lemmatizer(stem)

                tokens_result.append(lem)

        tokenized_data = finishing(tokens_result)

        str_data.append(stringify(tokenized_data))
        token_data.append(tokenized_data)

    return {
        "str_token": str_data,
        "tokens": token_data,
    }


# Term Frequency

@st.cache_data
def do_TF(corpus):
    # Memastikan input adalah DataFrame atau dictionary
    if isinstance(corpus, dict):
        if 'str_token' in corpus:
            corpus = pd.DataFrame(corpus)
        else:
            raise ValueError(
                "Dictionary input harus memiliki kunci 'str_token' dengan nilai yang tidak kosong.")
    elif not isinstance(corpus, pd.DataFrame):
        raise ValueError(
            "Input harus berupa DataFrame atau dictionary dengan kunci 'str_token'.")

    # Memastikan korpus berisi string yang valid
    if 'str_token' not in corpus.columns or corpus['str_token'].isnull().all():
        raise ValueError(
            "DataFrame input harus memiliki kolom 'str_token' dengan nilai yang tidak kosong.")

    # Menghapus dokumen yang kosong atau hanya berisi tanda baca
    corpus['str_token'] = corpus['str_token'].str.strip()
    corpus = corpus[corpus['str_token'].str.len() > 0]

    # Inisialisasi CountVectorizer dengan stop words bahasa Inggris
    count_vectorizer = CountVectorizer(analyzer='word', stop_words='english')

    try:
        # Fit dan transform korpus
        X = count_vectorizer.fit_transform(corpus['str_token'])
        count_tokens = count_vectorizer.get_feature_names_out()
    except ValueError as e:
        if "empty vocabulary" in str(e):
            raise ValueError(
                "Korpus input hanya berisi stop words atau kosong.")
        else:
            raise e

    return {
        "df": X,
        "features": count_tokens
    }


def create_TF_df(_term_freqs):
    return pd.DataFrame(data=_term_freqs['df'].toarray(), columns=_term_freqs['features'], index=['Abstrak 1', 'Abstrak 2'])


# create similarity dataframe
def create_sim(abs_ds):
    data_sim = np.zeros((abs_ds.shape[0], abs_ds.shape[0]))
    for d_utama_idx in range(1, abs_ds.shape[0]):
        for d_pembanding_idx in range(0, d_utama_idx):
            v_s = round(100 - (mean_absolute_percentage_error(abs_ds[d_utama_idx], abs_ds[d_pembanding_idx]) * 100), 1) if round(
                100 - (mean_absolute_percentage_error(abs_ds[d_utama_idx], abs_ds[d_pembanding_idx]) * 100), 1) >= 0 else 0
            data_sim[d_pembanding_idx][d_utama_idx] = v_s
            data_sim[d_utama_idx][d_pembanding_idx] = v_s
    return pd.DataFrame(data_sim, columns=['Abstrak 1', 'Abstrak 2'], index=['Abstrak 1', 'Abstrak 2'])


# App flow
st.title('Text Similarity using RBM & Term Frequency')

st.header('Masukkan Corpus')
with st.form('input_corpus'):
    text_1 = st.text_area(label='Masukkan abstrak 1',
                       placeholder='Masukkan abstrak yang akan dibandingkan')
    text_2 = st.text_area(label='Masukkan abstrak 2',
                       placeholder='Masukkan abstrak sebagai pembanding')
    submit = st.form_submit_button('Cek kesamaan')

if submit:
    corpus = np.array([text_1, text_2])
    preprocessed_corpus = preprocessing(corpus)

    count_vectorizer = CountVectorizer(binary=True)

    X = count_vectorizer.fit_transform(preprocessed_corpus["str_token"])
    y = count_vectorizer.get_feature_names_out()

    tf_df = create_TF_df({'df':X, 'features':y})
    
    # Train BernoulliRBM
    rbm = BernoulliRBM(n_components=X.shape[1], n_iter=10)
    abstracts_transformed = rbm.fit_transform(X)
    
    sim_df = create_sim(abstracts_transformed)

    st.header('Hasil')
    st.subheader('Matriks term frequency', divider='rainbow')
    st.dataframe(tf_df)
    st.subheader('Matriks kesamaan (%)', divider='rainbow')
    st.dataframe(sim_df)

