import numpy as np

# Main activation functions used in neural networks

def stepFunction(sum): #Problems that can be linearly separate
    if (sum >= 0):
        return 1
    return 0

def sigmoidFunction(sum): #Problems with binary classification
    return 1 / (1 + np.exp(-sum))

def tanhFunction(sum): # Problems betwenn -1 and 1
    return (np.exp(sum) - np.exp(-sum))/(np.exp(sum) + np.exp(-sum))

# High used function
def ReLU(sum): # used in CNN
    if (sum >=0):
        return sum
    return 0

def linearFunction(sum): # used in regression
    return sum

def softmaxFunction(x): # used in multi-classification
    ex = np.exp(x)
    return ex  /ex.sum()


print("Testing functions:")
print(" ")
print(stepFunction(5))
print(sigmoidFunction(5))
print(tanhFunction(5))
print(ReLU(5))
print(softmaxFunction(np.array([5,6,7,8,9])))
