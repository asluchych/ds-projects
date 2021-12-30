import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("ggplot")

x = np.linspace(0, 10, 1000)
y = np.sin(x)

fig = plt.figure()
ax = plt.axes()
ax.plot(x, y)
plt.show()

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Function sin(x)")
plt.show()

plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))
plt.show()

plt.plot(x, np.sin(x), color = "k")
plt.plot(x, np.cos(x), color = "r", linestyle = "--")
plt.show()


plt.plot(x, np.sin(x), "k:", label = "sin(x)")
plt.plot(x, np.cos(x), "r--", label = "cos(x)")
plt.legend()
plt.savefig("sincos.png")
plt.show()

# Scatterplot
presidents_df = pd.read_csv("president_heights_party.csv", index_col = "name")
plt.scatter(presidents_df["height"], presidents_df["age"])
plt.savefig("scatter.png")
plt.show()


plt.scatter(presidents_df["height"], presidents_df["age"], marker = "<", color = "b")
plt.xlabel("Height")
plt.ylabel("Age")
plt.title("US Presidents")
plt.savefig("scatter2.png")
plt.show()

# Plotting with Pandas
presidents_df.plot(kind = "scatter", x = "height", y = "age", title = "US Presidents")
plt.show()

# Histogram
presidents_df["height"].plot(kind = "hist", title = "Height", bins = 5)
plt.show()

plt.hist(presidents_df["height"], bins = 5)
plt.show()

# Boxplot
print(presidents_df["height"].describe())
plt.style.use("classic")
presidents_df.boxplot(column = "height")
plt.show()

# Bar plot
party_cnt = presidents_df["party"].value_counts()
plt.style.use("ggplot")
party_cnt.plot(kind = "bar")
plt.show()
