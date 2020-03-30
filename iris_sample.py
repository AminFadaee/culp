import numpy
from sklearn import datasets
from sklearn.model_selection import train_test_split

from culp.classifier import CULPUsingKNNFactory
from culp.link_predictors import CompatibilityScoreStrategy
from culp.similarity_strategies import VectorSimilarityMetric

iris = datasets.load_iris()
data = iris['data']
labels = iris['target']  # labels should be between 0 and C-1
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, random_state=42)
culp_classifier = CULPUsingKNNFactory.create(
    X_train,
    y_train,
    X_test,
    classes=[0, 1, 2],
    similarity=VectorSimilarityMetric.manhattan,
    k=11
)
culp_classifier.train()
prediction = culp_classifier.predict(CompatibilityScoreStrategy())
print("Prediction Accuracy = {0}%".format(round(100 * float(numpy.mean(prediction == y_test)), 2)))
