from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Train and save a simple model
iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier().fit(X, y)

with open("../model/model.pkl", "wb") as f:
    pickle.dump(model, f)