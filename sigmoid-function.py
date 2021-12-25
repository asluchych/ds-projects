import math

w1, w2, b, x1, x2 = [float(x) for x in input("Please enter w1, w2, b, x1 and x2 separated by spaces: ").split()]

node_out = 1/(1 + math.exp(-(w1*x1 + w2*x2 + b)))

print("The node output is:", round(node_out, 4))
