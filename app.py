# import streamlit as st
# import requests
# from bs4 import BeautifulSoup
# import gensim
# from gensim.models import Word2Vec
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.decomposition import PCA

# # Function to extract text from a URL
# def get_text_from_url(url):
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#     text = ' '.join([p.get_text() for p in soup.find_all('p')])
#     return text

# # Function to generate word embeddings and visualize
# def generate_embeddings(text):
#     # Tokenize the text
#     tokenized_text = [sentence.split() for sentence in text.split('.')]
#     # Train Word2Vec model
#     model = Word2Vec(tokenized_text, min_count=1)
#     # Get vocabulary
#     words = list(model.wv.index_to_key)
#     # Get word vectors
#     word_vectors = model.wv.vectors
#     # Apply PCA to reduce dimensionality for visualization
#     pca = PCA(n_components=2)
#     principal_components = pca.fit_transform(word_vectors)
#     # Create DataFrame for plotting
#     embeddings_df = pd.DataFrame(principal_components, columns=['x', 'y'])
#     embeddings_df['word'] = words

#     # Plot the embeddings
#     fig, ax = plt.subplots()
#     for word, x, y in zip(embeddings_df['word'], embeddings_df['x'], embeddings_df['y']):
#         ax.annotate(word, (x, y))
#     ax.scatter(embeddings_df['x'], embeddings_df['y'], alpha=0.5)
#     st.pyplot(fig)

#     return model

# # Streamlit UI
# st.title('Word Embeddings Visualization')

# input_type = st.radio('Select input type:', ('Sentence', 'URL'))

# if input_type == 'Sentence':
#     sentence = st.text_area('Enter a sentence:')
#     if st.button('Generate Embeddings'):
#         model = generate_embeddings(sentence)

# elif input_type == 'URL':
#     url = st.text_input('Enter a URL:')
#     if st.button('Extract Text and Generate Embeddings'):
#         text = get_text_from_url(url)
#         model = generate_embeddings(text)

import streamlit as st
import requests
from bs4 import BeautifulSoup
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np
import base64

# Function to extract text from a URL
def get_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = ' '.join([p.get_text() for p in soup.find_all('p')])
    return text

# Function to remove stopwords
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])

# Function to generate word embeddings and visualize
def generate_embeddings(text):
    # Remove stopwords
    text = remove_stopwords(text)
    # Tokenize the text
    tokenized_text = [sentence.split() for sentence in text.split('.')]
    # Train Word2Vec model
    model = Word2Vec(tokenized_text, min_count=1)
    
    # Extracting words from the text
    words = []
    for word in text.split():
        if len(word) > 1:  # Considering only words with more than one character
            words.append(word)
    
    # Filtering out words not in the model's vocabulary
    words = [word for word in words if word in model.wv]
    # Get word vectors for the extracted words
    word_vectors = model.wv[words]
    
    # Apply PCA to reduce dimensionality for visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(word_vectors)
    
    # Create DataFrame for plotting
    embeddings_df = pd.DataFrame(principal_components, columns=['x', 'y'])
    embeddings_df['word'] = words

    # Plot the embeddings using PCA
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # PCA visualization
    for word, x, y in zip(embeddings_df['word'], embeddings_df['x'], embeddings_df['y']):
        ax[0].annotate(word, (x, y))
    ax[0].scatter(embeddings_df['x'], embeddings_df['y'], alpha=0.5)
    ax[0].set_title('PCA Visualization')

    # Plot clusters of similar words
    embedding_clusters = []
    word_clusters = []
    for word in words:
        embeddings = []
        similar_words = []
        try:
            similar_words.append(word)
            embeddings.append(model.wv[word])
            for similar_word, _ in model.wv.most_similar(word, topn=4):
                if similar_word in words:
                    similar_words.append(similar_word)
                    embeddings.append(model.wv[similar_word])
            embedding_clusters.append(embeddings)
            word_clusters.append(similar_words)
        except KeyError:
            st.warning(f"No similar words found for '{word}'.")

    for embeddings, similar_words in zip(embedding_clusters, word_clusters):
        embeddings = np.array(embeddings)  # Convert to NumPy array
        ax[1].scatter(embeddings[:,0], embeddings[:,1], alpha=0.5)
        for word, embd in zip(similar_words, embeddings):
            ax[1].text(embd[0], embd[1], word, fontsize=8)
    ax[1].set_title('Word Clusters')

    # Save the Word2Vec model to a file
    model_file = "word2vec_model.bin"
    model.save(model_file)
    with open(model_file, "rb") as f:
        model_bytes = f.read()

    # Download link for the Word2Vec model file
    href = f'<a href="data:application/octet-stream;base64,{base64.b64encode(model_bytes).decode()}" download="{model_file}">Download Word2Vec Model</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.pyplot(fig)

    return model

# Streamlit UI
st.title('Word Embeddings Visualization')

input_type = st.radio('Select input type:', ('Direct Text', 'Hyperlink', 'Upload Document'))

if input_type == 'Direct Text':
    text = st.text_area('Enter text:')
    if st.button('Generate Embeddings'):
        model = generate_embeddings(text)

elif input_type == 'Hyperlink':
    url = st.text_input('Enter a URL:')
    if st.button('Generate Embeddings'):
        text = get_text_from_url(url)
        model = generate_embeddings(text)

elif input_type == 'Upload Document':
    uploaded_file = st.file_uploader("Upload a text document", type=['txt'])
    if uploaded_file is not None:
        text = uploaded_file.getvalue().decode("utf-8")
        if st.button('Generate Embeddings'):
            model = generate_embeddings(text)
