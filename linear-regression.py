import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


from sklearn.datasets import load_boston
boston_dataset = load_boston()

boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston["MEDV"] = boston_dataset.target

# Univariate Linear Regression
X = boston[["RM"]]
y = boston["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model1 = LinearRegression()
model1.fit(X_train, y_train)
print("Model: MEDV =", model1.intercept_.round(2), "+", model1.coef_[0].round(2), "*RM")