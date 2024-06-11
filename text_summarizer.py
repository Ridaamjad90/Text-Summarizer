#text_summarizer.py
import pandas as pd
import numpy as np

#For configs
import yaml

#For Fetching Text from the URL and it's processing
from langchain.document_loaders import UnstructuredURLLoader
from unstructured.cleaners.core import remove_punctuation,clean,clean_extra_whitespace
from langchain.docstore.document import Document

#For text summarization: using a hugging face model
from transformers import pipeline

#For text pre-processing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')


def count_words(text):
    words = text.split()
    print("word count of this article:")
    print(len(words))

def generate_document(config, url):
    if config['Input'][0]['type_of_input']['unstructured_url']:
        print("Fetch the text from the following URL:")
        print(config['Input'][1]['URL'])

        loader = UnstructuredURLLoader(urls = [config['Input'][1]['URL']], mode = "elements",
                                   post_processors=[clean,remove_punctuation,clean_extra_whitespace])
        elements = loader.load()
        selected_elements = [e for e in elements if e.metadata['category']=="NarrativeText"]
        full_clean = " ".join([e.page_content for e in selected_elements])
        return full_clean, Document(page_content=full_clean, metadata={"source":url})
    else:
        print("Fetch the text from the text.txt file")
        file_path = config['Input'][2]['File_path']
        with open(file_path, 'r') as file:
            file_contents = file.read()
        return file_contents, Document(page_content=file_contents, metadata={"source":file_path})
        

def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    return processed_text
    
def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = 'positive' if scores['pos'] > 0 else 'negative'
    print(sentiment)

def run_summary(config):
    #Generate Document
    x = generate_document(config, config['Input'][1]['URL'])
    full_clean = x[0]
    full_document = x[1]

    #Now count the total words in the article
    count_words(full_clean)

    #Now count the total characters in the article
    print("total character count of this article:")
    print(len(full_clean))

    #Load a hugging face text summarization model. You can also upload an  OpenAI model here 
    summarizer = pipeline("summarization", model="Falconsai/text_summarization")
    print("Here's the summary of the article on the provided text")
    print(summarizer(full_clean, max_length=config['Summarizer'][0]['max_length'], min_length=config['Summarizer'][1]['min_length'], do_sample=False))

    #PreProcess the text in order to run more sentiment analysis on it:
    print("Now running a detailed pre-processing on text: tokenization, removing stop words, lemmatization and converting to clean text")
    pre_processed_text = preprocess_text(full_clean)

    #Run Sentiment analysis on it:
    print("The Sentiment of this  Text is:")
    get_sentiment(pre_processed_text)
