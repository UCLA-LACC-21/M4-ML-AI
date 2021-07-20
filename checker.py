'''
Checker function for lab 1. 
Please don't see the code inside. If you don't know how to solve a problem, just ask the instructor.
'''
import numpy as np
import pandas as pd


def compute_values():
    np.random.seed(0)
    X = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5
    res = 0.5 * np.random.randn(100)       # Generate 100 residual terms
    y = 2 + 0.3 * X + res                  # Actual values of Y
    
    # Create pandas dataframe to store our X and y values
    df = pd.DataFrame(
        {'X': X,
         'y': y}
    )
    # Calculate the mean of X and y
    xmean = np.mean(X)
    ymean = np.mean(y)
    
    # Calculate the terms needed for the numator and denominator of beta
    df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)
    df['xvar'] = (df['X'] - xmean)**2
    
    # Calculate beta and alpha, no multi-dimentional matrix operations here since we are processing with 1d data.
    beta = df['xycov'].sum() / df['xvar'].sum()
    alpha = ymean - (beta * xmean)
    ypred = alpha + beta * X
    
    class Perceptron_check(object):
        def __init__(self, input_size, lr=1, epochs=100):
            self.W = np.zeros(input_size+1)
            # add one for bias
            self.epochs = epochs
            self.lr = lr

        def activation_fn(self, x):
            #return (x >= 0).astype(np.float32)
            return 1 if x >= 0 else 0
    
        def predict(self, x):
            z = self.W.T.dot(x)
            a = self.activation_fn(z)
            return a
    
        def fit(self, X, d):
            for _ in range(self.epochs):
                for i in range(d.shape[0]):
                    x = np.insert(X[i], 0, 1)
                    y = self.predict(x)
                    e = d[i] - y
                    self.W = self.W + self.lr * e * x

        def inference(self, X):
            ypred = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                x = np.insert(X[i], 0, 1)
                ypred[i] = self.predict(x)
            return ypred
        '''
        Now we have finished our own perceptron implementation, we now perform a simple test to verify it.
        '''
        
    #Let's generate some random input and target data
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 1]
    ])
    #y = np.array([0, 0, 0, 0, 1])
    y = np.array([0, 1, 1, 1, 1])
    #Then call our implemented perceptron class
    perceptron = Perceptron_check(input_size=2)
    perceptron.fit(X, y)
    return alpha, ypred, perceptron.W
        
def check_alpha(alpha_check):
    alpha, ypred, W = compute_values()
    if alpha_check == alpha:
        print("Congratulations! Your implementation is correct! You can move forward to the next block!")
    else:
        print("Unfortunately the implementation is not correct, pls try again. If you don't know how to implement it you can ask instructor for help")
        
def check_ypred(ypred_check):
    alpha, ypred, W = compute_values()
    if np.array_equal(ypred_check,ypred):
        print("Congratulations! Your implementation is correct! You can move forward to the next block!")
    else:
        print("Unfortunately the implementation is not correct, pls try again. If you don't know how to implement it you can ask instructor for help")
        
def check_perceptron(W_check):
    alpha, ypred, W = compute_values()
    if np.array_equal(W,np.array(W_check)):
        print("Congratulations! Your implementation is correct! You can move forward to the next block!")
    else:
        print("Unfortunately the implementation is not correct, pls try again. If you don't know how to implement it you can ask instructor for help")
    