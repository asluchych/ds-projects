import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_boston

from sklearn.metrics import mean_squared_error

# Load boston dataset
boston_dataset = load_boston()

# Convert data into a pandas DataFrame
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# Add price column (a target value in the dataset) to the DataFrame
boston["MEDV"] = boston_dataset.target

# Exploratory data analysis
print('\n')
print("The shape of the data frame is: ", boston.shape)
print("The colums are : ", boston.columns)
print('\n')
print("Summary statistics:\n", boston.describe().round(2))
print('\n')
print(boston[['MEDV', 'RM']].head(10))
print('\n')
boston.hist(column = 'RM', bins = 20)
plt.savefig('plot1.png')
plt.show()

# Correlation matrix
corr_matrix = boston.corr().round(2)
print('\n')
print('The correlation matrix is:\n', corr_matrix)
print('\n')

# Data preparation: feature selection
boston.plot(kind = 'scatter', x = 'RM', y = 'MEDV', figsize = (8,6))
plt.savefig('plot1.png')
plt.show()

boston.plot(kind = 'scatter', x = 'LSTAT', y = 'MEDV', figsize = (8, 6))
plt.savefig('plot2.png')
plt.show()

# Univariate Linear Regression
# Use the number of rooms as a feature matrix and the price as a target
X = boston[["RM"]]
y = boston["MEDV"]

# Make a train test split with a test size of 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model1 = LinearRegression()
model1.fit(X_train, y_train)
print("Model: MEDV =", model1.intercept_.round(2), "+", model1.coef_[0].round(2), "*RM")

# Model evaluation
# Draw a fitted line
y_test_predicted = model1.predict(X_test)
plt.scatter(X_test, y_test, label = 'Testing Data')
plt.plot(X_test, y_test_predicted, label = 'prediction', linewidth = 3)
plt.xlabel("RM")
plt.ylabel('MEDV')
plt.legend(loc = 'upper left')
plt.savefig("fitted.png")
plt.show()

# Draw residuals vs fitted
residuals = y_test - y_test_predicted
plt.scatter(y_test_predicted, residuals, label = 'Residuals vs. Fitted')
plt.hlines(y = 0, xmin = y_test_predicted.min(), xmax = y_test_predicted.max(), linestyle = "--")
plt.legend(loc = 'upper left')
plt.xlabel("Fitted")
plt.ylabel("Residuals")
plt.savefig("residuals_fitted.png")
plt.show()

# Evaluating the model
print("\n")
print("MSE of the model is: ", mean_squared_error(y_test, y_test_predicted))
print("R-squared of the model is: ", model1.score(X_test, y_test))
print("\n")

# Make a prediction
new_RM = int(input("Please enter the number of rooms in a house which price you want to predict: "))
new_RM = np.array([new_RM]).reshape(-1, 1)
print("The price of a house with {0} rooms is predicted to be {1} USD".format(new_RM[0, 0], (1000*model1.predict(new_RM)[0]).round(2)))

# Multivariate Linear Regression
# Use the number of rooms and the percentage of lower status in population as a feature matrix and the price as a target
X = boston[["RM", "LSTAT"]]
y = boston["MEDV"]

# Make a train test split with a test size of 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model2 = LinearRegression()
model2.fit(X_train, y_train)
print("Multivariate model: MEDV =", model2.intercept_.round(2), "+", model2.coef_[0].round(2), "*RM", "+", model2.coef_[1].round(2), "*LSTAT")


# Draw residuals vs fitted
y_test_predicted = model2.predict(X_test)
residuals = y_test - y_test_predicted
plt.scatter(y_test_predicted, residuals, label = 'Residuals vs. Fitted')
plt.hlines(y = 0, xmin = y_test_predicted.min(), xmax = y_test_predicted.max(), linestyle = "--")
plt.legend(loc = 'upper left')
plt.xlabel("Fitted Mulitvariate")
plt.ylabel("Residuals Multivariate")
plt.savefig("residuals_fitted2.png")
plt.show()

# Evaluating the model
print("\n")
print("MSE of the multivariate model is: ", mean_squared_error(y_test, y_test_predicted))
print("R-squared of the multivariate model is: ", model2.score(X_test, y_test))
