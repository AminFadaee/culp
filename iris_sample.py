from culp.classifier import culp
import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
data = iris['data']
labels = iris['target']  # labels should be between 0 and C-1
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42)
prediction = culp(X_train, y_train, X_test, link_predictor='CS', similarity='manhattan', k=11)
print("Prediction Accuracy = {0}%".format(round(100 * float(numpy.mean(prediction == y_test)), 2)))
