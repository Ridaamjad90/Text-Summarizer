# Text-Summarizer

This text summarizer comprises of a yaml config file, a py file and an ipynb file that can be used to leverage an LLM model to summarize a text.

### Text Input options:
1. In the config file, you have an option to set the unstructured_url = True and a URL that you can input. It will fetch the written content from that URL
2. if unstructured_url is False, then it will fetch the read.txt file from your main folder and use that for summarization.

### Text Summarizer:
1. Right now, i have used a hugging face text summarizer model: Falconsai/text_summarization. Although it can be amended to use OpenAI GPT3.5 models too (it requires API Key and can work based on openAI's price plans)
2. You can set max and min words of the summary

### Text EDA: You also get the following:
1. word count of the article
2. total character counts of the article
3. Sentiment of the article: for this I use nltk's SentimentIntensityAnalyzer

