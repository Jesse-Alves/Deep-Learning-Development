import numpy as np

# Main activation functions used in neural networks

def stepFunction(sum):
    if (sum >= 0):
        return 1
    return 0

def sigmoidFunction(sum):
    return 1 / (1 + np.exp(-sum))

def tanhFunction(sum):
    return (np.exp(sum) - np.exp(-sum))/(np.exp(sum) + np.exp(-sum))


print("Testing functions:")
print(" ")
print(stepFunction(5))
print(sigmoidFunction(5))
print(tanhFunction(5))