Word Embeddings Visualization
This is a Streamlit web application that generates and visualizes word embeddings using Word2Vec. It allows users to input text directly, extract text from a hyperlink, or upload a text document. The application provides insights into the semantic relationships between words in the input text.

Features
Input Options: Users can choose between three input options: direct text input, hyperlink input, or uploading a text document.
Word Embeddings: The application generates word embeddings using Word2Vec based on the input text.
Visualization: Word embeddings are visualized using Principal Component Analysis (PCA) to reduce dimensionality. Additionally, clusters of similar words are plotted for better understanding of semantic relationships.
Stopword Removal: Common English stopwords are removed from the input text to focus on meaningful content.
Web Scraping: Text content can be extracted from a hyperlink using web scraping techniques.

Installation
Clone the repository:
git clone https://github.com/yourusername/word-embeddings-visualization.git

Navigate to the project directory:
cd word-embeddings-visualization
Install the required dependencies:

pip install -r requirements.txt

Usage
Run the Streamlit app:

streamlit run app.py
Access the application in your web browser at the provided URL.

Choose the input type (direct text, hyperlink, or upload document) and follow the prompts to generate and visualize word embeddings.
