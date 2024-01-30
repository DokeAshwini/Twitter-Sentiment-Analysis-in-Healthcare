# A Comparison of Naïve Bayes and Decision Tree Approaches to Twitter Sentiment Analysis in Healthcare as a Binary Classification Problem   

## Project Overview
This project analyzes the health-related tweet from 16 different publishers to derive insights about the health-related news articles collected from https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter. The project involves loading the tweet data into a Spark DataFrame, preprocessing the data by tokenizing and stemming the words, and then performing sentiment analysis on the tweets using the VADER sentiment analysis library from NLTK. Following that, visualisation is carried out to acquire some understanding of the sentiments according to the year, months, and publications. Finally, the project uses PySpark's machine learning libraries to train and assess machine learning models (Naive Bayes and Decision Tree classifiers) to categorise tweets based on their sentiment (positive or negative).  The models are evaluated using cross-validation and MulticlassClassificationEvaluator, and the results are displayed using Matplotlib and Seaborn in a confusion matrix. 

## Files in the repository
**data directory**: contains 16 .txt files that hold the health-related tweets from  16 different publishers.  
**README.md**: provides an overview of the project.  
**health_sentiment_analysis.ipynb**: IPython Notebook file containing the code for the project.

## Dependencies
PySpark  
Pandas  
Matplotlib  
Seaborn  
nltk  
re  

## Results
The analysis performed in this project revealed that there are significant differences in the health-related news articles published by different publishers. Additionally, the machine learning model trained on the preprocessed data achieved an accuracy of 85% in predicting the sentiments of healthcare related tweets. Naïve Bayes outperformed the decision tree in F1 score and efficiency. The decision tree showed bias towards positive predictions and overfitting.