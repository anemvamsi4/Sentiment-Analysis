# Sentiment Analysis on Twitter Data

## Overview

Welcome to the Sentiment Analysis project repository! In this project, we tackle the task of classifying sentiments in Twitter messages. Our goal is to categorize tweets as positive, negative, or neutral using various techniques and language models.

## Dataset

We leverage the Sentiment140 dataset obtained from Kaggle, which comprises a staggering 1.6 million tweets, each labeled with a corresponding sentiment. This dataset serves as the foundation for our sentiment analysis experiments.

## Text Preprocessing

Cleaning and preparing the text data is a critical step. We explore different methods for text preprocessing, including using the NLTK and SpaCy libraries for text cleaning.

## Sentiment Analysis Techniques and Models

This project employs a variety of sentiment analysis techniques and models:

1. **Vader Sentiment Analysis**: We use the Vader sentiment analysis model, a rule-based tool, to analyze the sentiment of the preprocessed text data.

2. **Transformers Pipeline**: Leveraging the Transformers library, we utilize pre-trained transformer models like BERT to conduct sentiment analysis on the cleaned text data.

3. **RoBERTa Model**: Our project integrates the RoBERTa model, a state-of-the-art transformer-based language model. We fine-tune RoBERTa on the Sentiment140 dataset to enhance sentiment classification.

4. **Custom LSTM Model**: A custom Long Short-Term Memory (LSTM) model is implemented for sequential sentiment analysis. It learns to classify sentiment based on the textual context.

5. **Bag-of-Words with Naive Bayes and XGBoost**: We explore traditional machine learning approaches using the bag-of-words representation combined with Naive Bayes classifier and XGBoost for sentiment analysis on the cleaned text data.

## Project Structure

- `sentiment_analysis.ipynb`: Python notebook containing all the sentiment analysis techniques and models.
- `cleaned_text_data.csv`: CSV file containing the text data processed using NLTK and SpaCy.

## Conclusion

This project offers an in-depth exploration of sentiment analysis techniques, ranging from rule-based methods to advanced transformer models and traditional machine learning algorithms. It provides valuable insights into the performance and effectiveness of different approaches for sentiment classification on the Sentiment140 dataset.

Feel free to explore the code and experiments, and don't hesitate to reach out if you have any questions or feedback!
