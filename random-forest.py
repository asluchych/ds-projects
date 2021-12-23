import numpy as np

random_state = int(input("Please enter a random state: "))
n = int(input("Please enter a number of rows: "))

rows = []
for i in range(n):
    rows.append([float(a) for a in input("Please enter a feature matrix row separated by spaces: ").split()])

X = np.array(rows)
y = np.array([int(a) for a in input("Please enter target values: ").split()])

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = random_state)

rf = RandomForestClassifier(n_estimators = 5, random_state = random_state)
rf.fit(X_train, y_train)

print("Random Forest predictions are: ", rf.predict(X_test))


