from traditional_methods import get_data, traditional
from sklearn.linear_model import LogisticRegression

data = get_data.read_file("labeled_data.csv")
logistic_regression_method = traditional.Traditional_method(LogisticRegression(max_iter=1000), "logistic regression")
logistic_regression_method.classification_report(get_data.upsampling(data))