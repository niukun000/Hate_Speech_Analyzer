import collections
import nltk
import os
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC

# Define some stop words
import pandas as pd

stop_words = {
    'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 
    'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
    'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 
    'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as',
    'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',
    'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more',
    'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above',
    'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any',
    'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
    'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can',
    'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where',
    'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't',
    'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how',
    'further', 'was', 'here', 'than'}


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
    X = f["tweet"]
    Y = f["class"]
    return X,Y


def preprocess_sentence(sentence):
    lemmatizer = nltk.WordNetLemmatizer()
    # clearly list out our preprocessing pipeline
    processed_tokens = nltk.word_tokenize(sentence)
    processed_tokens = [w.lower() for w in processed_tokens]
    # find least common elements
    word_counts = collections.Counter(processed_tokens)
    uncommon_words = word_counts.most_common()[:-10:-1]
    # remove these tokens
    processed_tokens = [w for w in processed_tokens if w not in stop_words]
    processed_tokens = [w for w in processed_tokens if w not in uncommon_words]
    # lemmatize
    processed_tokens = [lemmatizer.lemmatize(w) for w in processed_tokens]
    return processed_tokens


def feature_extraction(tokens):
    '''Turn each word into a feature. The feature value is the word count.'''
    return dict(collections.Counter(tokens))


def train_test_split(dataset, train_size=0.8):
    num_training_examples = int(len(dataset) * train_size)
    return dataset[:num_training_examples], dataset[num_training_examples:]


examples = load_files("labeled_data.csv")
# all_examples = [(preprocess_sentence(example[0]), example[1]) for example in examples]
i = 0
all_examples = []
for i in range(len(examples[0])):
    all_examples.append((preprocess_sentence(examples[0][i]), examples[1][i]))
print(all_examples)
random.shuffle(all_examples)
#
# print('{} emails processed.'.format(len(all_examples)))

featurized = [(feature_extraction(corpus), label)
              for corpus, label in all_examples]
#
training_set, test_set = train_test_split(featurized, train_size=0.7)

classif = SklearnClassifier(LinearSVC())

# model = nltk.classify.NaiveBayesClassifier.train(training_set)
model = classif.train(training_set)
training_error = nltk.classify.accuracy(model, training_set)
print('Model training complete. Accuracy on training set: {}'.format(
    training_error))

testing_error = nltk.classify.accuracy(model, test_set)
print('Accuracy on test set: {}'.format(testing_error))

# y_pred = model.predict()
# print(metrics.classification_report(
#     y_test, y_pred, target_names=["hate_speech", "offensive_language", "neither"]))