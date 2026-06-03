from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = data.data
y = data.target
print(X.shape)
print(y.shape)