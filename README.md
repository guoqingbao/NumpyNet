# NumpyNet
Neural network implemented with Numpy in ~100 lines of code

A feed-forward neural network implemented with Numpy achieved decent performance (~98% accuracy on MNIST dataset) compared to state-of-the-arts.

### YouTube link: https://youtu.be/Jp7fR1iP-ew

SGD with Mini-batch and learning rate decay are also implemented.

### CPU usage: 6-8% in AMD R7 5800H (8 cores)

### Init network weights
```python
    def create(self): # init network weights
        for i in range(self.n_weights):
            self.params['W'+str(i+1)] = np.random.randn(self.nodes[i+1], self.nodes[i]) * np.sqrt(1. / self.nodes[i+1]) #multiply with small values
```


_A 2 hidden-layer implementation can be found in example folder._ 
### Forward and backward propagations
```python
    def forward(self, x):
        out = x
        self.params['O0'] = out
        for i in range(self.n_weights):
            order = i + 1
            self.params['Z'+str(order)] = np.dot(self.params['W'+str(order)], out)
             
            if order==self.n_weights: #last layer softmax
                out = self.softmax(self.params['Z'+str(order)])
            else: #other layers with ReLU
                out = self.relu(self.params['Z'+str(order)])
            self.params['O'+str(order)] = out
        return out
    
    def backward(self, y, output):
        wparams = {}
        #the last layer
        error = 2 * (output - y)/output.shape[0] * self.softmax(self.params['Z'+str(self.n_weights)], deriv=True)
        wparams['W'+str(self.n_weights)] = np.outer(error, self.params['O'+str(self.n_weights-1)])

        #the former layers
        for i in range(1, self.n_weights):
            error = np.dot(self.params['W'+str(self.n_weights-i+1)].T, error) * self.relu(self.params['Z'+str(self.n_weights-i)], deriv=True)
            wparams['W'+str(self.n_weights-i)] = np.outer(error, self.params['O'+str(self.n_weights-i-1)])
            
        return wparams
```
### Update network weight with changes calculated in backward pass
```python
    def update(self, wparams, lr):
        for key, v in wparams.items():
            self.params[key] -= lr * v 
```
### Training with SGD
```python
    def train_(self, x_train, y_train, x_test, y_test):
        for iteration in range(self.epochs):
            for x, y in zip(x_train, y_train):
                output = self.forward(x)
                wparams = self.backward(y, output)
                self.update(wparams)
                
            acc = self.accuracy(x_test, y_test)
            print("Epoch {0} -- Accuracy {1:.05f}%".format(iteration, acc))
```
### More advanced training (mini batch, learning rate decay)
```python
    def train(self, x_train, y_train, x_test, y_test, batch_size=1, lr_decay = False):
        if batch_size < 2: #conventional SGD
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
```

### Create model and traing the network
```python
model = NumpyNet(nodes = [784, 128, 64, 10], epochs = 20, lr = 0.5)
model.train(x_train,y_train,x_test,y_test, batch_size = 4, lr_decay = True)
```

### Prediction
```python
np.argmax(model.forward(x_test[110])) #predict the class of a test image
plt.imshow(x_test[110].reshape(28, 28), cmap='gray') #visualize the image

```
