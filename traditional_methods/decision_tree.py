from sklearn.tree import DecisionTreeClassifier
from traditional_methods import get_data, traditional

data = get_data.read_file("labeled_data.csv")
decision_tree_method = traditional.Traditional_method( DecisionTreeClassifier(), "DecisionTreeClassifier")
decision_tree_method.classification_report(get_data.upsampling(data), "upsampling")