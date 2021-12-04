import numpy as np

n, p = [int(x) for x in input("Please enter the number of rows and the number of columns of a matrix separated by spaces: ").split()]

X = []

for i in range(n):
    X.append([float(x) for x in input("Enter values of the row in the feature matrix separated by spaces: ").split()])

y = [float(x) for x in input("Enter values of target separated by spaces: ").split()]

X = np.array(X).reshape(n, p)
y = np.array(y)

matr_prod1 = np.matmul(np.transpose(X), X) #calculate X'X
matr_prod2 = np.matmul(np.transpose(X), y) #calculate X'y
b = np.matmul(np.linalg.inv(matr_prod1), matr_prod2) #calculate OLS coefficients (X'X)^(-1)*X'y
print("Coefficients of the OLS regression are: ", np.round_(b, 2))