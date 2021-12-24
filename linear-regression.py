import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split



from sklearn.datasets import load_boston

# Load boston dataset
boston_dataset = load_boston()

# Convert data into a pandas DataFrame
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# Add price column (a target value in the dataset) to the DataFrame
boston["MEDV"] = boston_dataset.target

# Univariate Linear Regression
# Use a number of rooms as a feature matrix and the price as a target
X = boston[["RM"]]
y = boston["MEDV"]

# Make a train test split with a test size of 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model1 = LinearRegression()
model1.fit(X_train, y_train)
print("Model: MEDV =", model1.intercept_.round(2), "+", model1.coef_[0].round(2), "*RM")

# Make a prediction
new_RM = int(input("Please enter the number of rooms in a house which price you want to predict: "))
new_RM = np.array([new_RM]).reshape(-1, 1)
print("The price of a house with {0} rooms is predicted to be {1} USD".format(new_RM[0, 0], (1000*model1.predict(new_RM)[0]).round(2)))