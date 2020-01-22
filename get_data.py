import re

import pandas as pd
from sklearn.utils import resample


def clean_text(self, df, text_field="tweet"):
    '''

    :param df: panda data frame
    :param text_field: string
    :return: panda data frame
    '''
    df[text_field] = df[text_field].str.lower()
    df[text_field] = df[text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", elem))
    return df


def read_file(file):
    '''
    read csv file
    :param file: csv file
    :return: a triple of datasets
    '''
    df = pd.read_csv(file)
    return df


def downsampling(data):
    '''
    restructure dataset to make the number of items in each categary in the set balance
    :param data: panda dataframe
    :return: panda dataframe
    '''
    hate_speech = data[data["class"] == 0]
    offensive_language = data[data["class"] == 1]
    neither = data[data["class"] == 2]
    neither = resample(neither,
                       replace=True,
                       n_samples=len(hate_speech),
                       random_state=123)
    offensive_language = resample(offensive_language,
                                  replace=True,
                                  n_samples=len(hate_speech),
                                  random_state=123)
    downsampled = pd.concat([hate_speech, offensive_language, neither])
    downsampled["class"].value_counts()
    return downsampled


def upsampling(data):
    '''
    restructure dataset to make the number of items in each categary in the set balance
    :param data: panda data frame
    :return: panda data frame
    '''
    hate_speech = data[data["class"] == 0]
    offensive_language = data[data["class"] == 1]
    neither = data[data["class"] == 2]
    hate_speech = resample(hate_speech,
                           replace=True,
                           n_samples=len(offensive_language),
                           random_state=123)
    neither = resample(neither,
                       replace=True,
                       n_samples=len(offensive_language),
                       random_state=123)
    upsampled = pd.concat([hate_speech, offensive_language, neither])
    upsampled["class"].value_counts()
    return upsampled