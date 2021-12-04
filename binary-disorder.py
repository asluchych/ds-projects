y_true = [x for x in input("Please enter zeros and ones separated by spaces for true values: ").split()]
y_pred = [x for x in input("Please enter the same number of zeros and ones separated by spaces for predicted values: ").split()]


import numpy as np

y_true = np.array(y_true)
y_pred = np.array(y_pred)

from sklearn.metrics import confusion_matrix
print("\n The confusion matrix is: \n", confusion_matrix(y_pred, y_true, labels=["1", "0"])/1 ) 
