import itertools
import random

import numpy as np
import nltk
import pandas as pd
from sklearn import (
    datasets, feature_extraction, model_selection, pipeline,
    svm, metrics
)
import matplotlib.pyplot as plt


def extract_features(corpus):
    '''Extract TF-IDF features from corpus'''

    stop_words = nltk.corpus.stopwords.words("english")

    # vectorize means we turn non-numerical data into an array of numbers
    count_vectorizer = feature_extraction.text.CountVectorizer(
        lowercase=True,  # for demonstration, True by default
        tokenizer=nltk.word_tokenize,  # use the NLTK tokenizer
        min_df=2,  # minimum document frequency, i.e. the word must appear more than once.
        ngram_range=(1, 2),
        stop_words=stop_words
    )
    processed_corpus = count_vectorizer.fit_transform(corpus)
    processed_corpus = feature_extraction.text.TfidfTransformer().fit_transform(
        processed_corpus)

    return processed_corpus
def load_files(directory):
    # result = []
    # for fname in os.listdir(directory):
    #     with open(directory + '/' + fname, 'r', encoding='ISO-8859-1') as f:
    #         result.append(f.read())
    f = pd.read_csv(directory)
    f.dropna()
    f = f.loc[:,["tweet", "class"]]
    f.sample(frac=1)
    # print(f)
    # random.shuffle(f)
    X = f["tweet"]
    Y = f["class"]

    return X,Y
if __name__ == '__main__':
    newsgroups_data= load_files("labeled_data.csv")
    #     = \
    #     datasets.load_files(
    #     '20_newsgroups', shuffle=True, random_state=42, encoding='ISO-8859-1')
    #
    # print('Data loaded.\nClasses = {classes}\n{datapoints}'.format(
    #     classes=newsgroups_data.target_names,
    #     datapoints=len(newsgroups_data.data)))

    # print(newsgroups_data.data[0])
    # random.shuffle(newsgroups_data)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        newsgroups_data[0], newsgroups_data[1], test_size=0.33,
        random_state=42)
    print(X_test)

    model = pipeline.Pipeline([
        ('counts', feature_extraction.text.CountVectorizer()),
        ('tfidf', feature_extraction.text.TfidfTransformer()),
        ('svm', svm.LinearSVC()),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print('Accuracy of SVM= {}'.format(
        np.mean(y_pred == y_test)))

    print(metrics.classification_report(
        y_test, y_pred, target_names=["hate_speech","offensive_language","neither"]))
