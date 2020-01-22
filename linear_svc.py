from sklearn.svm import LinearSVC
from traditional_methods import get_data, traditional

data = get_data.read_file("labeled_data.csv")
upsampled_data = get_data.upsampling(data)
downsampled_data = get_data.downsampling(data)

linear_svc = traditional.Traditional_method(LinearSVC(), "LinearSVC")

linear_svc.classification_report(data)
linear_svc.classification_report(upsampled_data)
linear_svc.classification_report(downsampled_data)