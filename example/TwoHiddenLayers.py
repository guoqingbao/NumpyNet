#!/usr/bin/env python
# coding: utf-8

# ## Library Prerequisites
import random
import numpy as np
np.random.seed(42)
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


# ## One-hot Encoding

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# ## Preprocessing

x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

x = (x/255.).astype(np.float32)
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)


plt.imshow(x_train[0].reshape(28, 28), cmap='gray')


# ## Network Implementation

class NumpyNet():
    def __init__(self, nodes, epochs=20, lr=0.5):
        self.nodes = nodes # neurons in each layer
        self.epochs = epochs
        self.lr = lr # learning rate
        self.params = {}
        self.create()
        
    def create(self): # init network weights
        params = self.params
        # three network weights
        params['W1'] = np.random.randn(self.nodes[1], self.nodes[0]) * np.sqrt(1./ self.nodes[1]) #multiply with small values
        params['W2'] = np.random.randn(self.nodes[2], self.nodes[1]) * np.sqrt(1./ self.nodes[2])        
        params['W3'] = np.random.randn(self.nodes[3], self.nodes[2]) * np.sqrt(1./ self.nodes[3])   
    
    def relu(self, x, deriv = False):
        if deriv:
            return x > 0 #derivative of relu
        return np.maximum(0, x)
    
    def softmax(self, x, deriv = False):
        exps = np.exp(x - x.max())
        if deriv:
            return exps/np.sum(exps, axis=0) * (1- exps/np.sum(exps, axis=0)) #derivative of softmax
        return exps / np.sum(exps, axis=0)
    
    def forward(self, x):#forward path
        params = self.params
        params['O0'] = x
        params['Z1'] = np.dot(params['W1'], x)
        params['O1'] = self.relu(params['Z1']) #input --> hidden layer 1
        
        params['Z2'] = np.dot(params['W2'], params['O1'])
        params['O2'] = self.relu(params['Z2']) #hidden layer 1 -> 2
        
        params['Z3'] = np.dot(params['W3'],params['O2'])
        params['O3'] = self.softmax(params['Z3']) #hidden layer 2 -> 3
        return params['O3']
    
    def backward(self, y, output):
        wparams = {}
        params = self.params
        error = 2 * (output - y)/output.shape[0] * self.softmax(params['Z3'], deriv=True)
        wparams['W3'] = np.outer(error, params['O2']) #weight changes for W3
        
        error = np.dot(params['W3'].T, error) * self.relu(params['Z2'], deriv=True)
        wparams['W2'] = np.outer(error, params['O1']) #weight changes for W2
        
        error = np.dot(params['W2'].T, error) * self.relu(params['Z1'], deriv=True)
        wparams['W1'] = np.outer(error,  params['O0'])#weight changes for W1
        
        return wparams
    
    def update(self, wparams, lr):
        for key, v in wparams.items():
            self.params[key] -= lr * v #update network weight with changes calculated before
            
    def accuracy(self, x_test, y_test):
        predictions = []
        for x, y in zip(x_test, y_test):
            output = self.forward(x)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        return np.mean(predictions)
    
    def train_(self, x_train, y_train, x_test, y_test):#train with SGD
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward(x)
                wparams = self.backward(y, output)
                self.update(wparams)
                
            acc = self.accuracy(x_test, y_test)
            print("Epoch {0} -- Accuracy {1:.05f}%".format(iteration, acc))
            
    def train(self, x_train, y_train, x_test, y_test, batch_size=1, lr_decay = False):#train with mini-batch SGD
        if batch_size < 2:
            return self.train_(x_train, y_train, x_test, y_test)
        
        for iteration in range(self.epochs):
            traindata = np.concatenate((x_train, y_train), axis=1)
            random.shuffle(traindata)
            batches = [traindata[k:k+batch_size] for k in range(0, len(traindata), batch_size)] # mini batch
            
            lr = self.lr* (self.epochs-iteration)/self.epochs if lr_decay else self.lr # learning rate decay
            for batch in batches:
                wparams_ = defaultdict(list)
                wparams = {}
                for data in batch:
                    x, y = data[:784], data[784:]
                    output = self.forward(x)
                    wps = self.backward(y, output)
                    for key, v in wps.items():
                        wparams_[key].append(v)
                        
                for key, v in wparams_.items():        
                    wparams[key] = np.mean(wparams_[key],axis=0)#mean of mini batch weight changes
                self.update(wparams, lr)  # update weights in each mini batch
                
            acc = self.accuracy(x_test, y_test)
            print("Epoch {0} -- LR {1:.05f} -- Accuracy {2:.03f}%".format(iteration,lr, acc*100))          
    


model = NumpyNet(nodes = [784, 128, 64, 10], epochs = 20, lr = 0.5)
model.train(x_train,y_train,x_test,y_test, batch_size = 4, lr_decay = True)



plt.imshow(x_test[110].reshape(28, 28), cmap='gray') #test a image


np.argmax(model.forward(x_test[110])) #predict the class of that image

