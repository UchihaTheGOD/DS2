import pandas as pd
from sklearn.preprocessing import LabelEncoder
iris = pd.read_csv("Iris.csv")
print(iris)
le = LabelEncoder()
iris['code'] = le.fit_transform(iris['Species'])
print("\nDataset after Label Encoding:")
print(iris)
