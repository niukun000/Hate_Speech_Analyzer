from sklearn.naive_bayes import GaussianNB
from traditional_methods import traditional, get_data
import numpy as np
data = get_data.read_file("labeled_data.csv")
gaussion_nb_method = traditional.Traditional_method( GaussianNB(), "GaussionNB")
gaussion_nb_method.classification_report(get_data.upsampling(data))