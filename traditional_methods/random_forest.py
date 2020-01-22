from traditional_methods import get_data, traditional
from sklearn.ensemble import RandomForestClassifier
import time
data = get_data.read_file("labeled_data.csv")
random_forest_method = traditional.Traditional_method(RandomForestClassifier(), "RandomForestClassifier")

# random_forest_method.classification_report(data)

random_forest_method.classification_report(get_data.upsampling(data), "upsampling")



# random_forest_method.classification_report(get_data.downsampling(data), "downsampling")
