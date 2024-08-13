# text_utils.py
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tokenize_text(text, max_tokens=512):
    sentences = sent_tokenize(text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    
    token_chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence_tokens in tokens:
        if current_length + len(sentence_tokens) > max_tokens:
            token_chunks.append(current_chunk)
            current_chunk = sentence_tokens
            current_length = len(sentence_tokens)
        else:
            current_chunk.extend(sentence_tokens)
            current_length += len(sentence_tokens)
    
    if current_chunk:
        token_chunks.append(current_chunk)
    
    return token_chunks

def extract_keywords_nltk(text, num_keywords=5):
    # Tokenizar el texto
    tokens = word_tokenize(text.lower())
    # Eliminar stop words
    stop_words = set(stopwords.words('spanish'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Contar la frecuencia de las palabras
    word_freq = Counter(filtered_tokens)

    # Obtener las palabras más comunes como palabras clave
    keywords = [word for word, _ in word_freq.most_common(num_keywords)]
    return keywords

def preprocess_query(self, query, language='english'):
    tokens = word_tokenize(query, language=language)
    tokens = [word.lower() for word in tokens if word.isalnum()]

    # Obtener stopwords según el idioma
    stop_words = set(stopwords.words(language))
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)