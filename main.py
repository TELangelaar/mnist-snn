import os

import numpy as np
from mnist_datasets import MNISTLoader


## == Load training dataset == ##
def load_data():
    loader = MNISTLoader()
    images, labels = loader.load() 
    assert len(images) == 60000 and len(labels) == 60000

    assert images.shape == (60000, 784) ## 60.000 28x28 flattened grayscale images of handwritten digits
    assert labels.shape == (60000,) ## 10 class labels (0-9)

    # Load test dataset
    images_test, labels_test = loader.load(train=False)
    assert len(images_test) == 10000 and len(labels_test) == 10000

    images_train = images.T  # Transpose to shape (784, 60000)
    labels_train = labels.T  # Transpose to shape (60000,)
    images_test = images_test.T
    labels_test = labels_test.T

    images_train = images_train / 255 # normalize
    images_test = images_test / 255

    return images_train, one_hot(labels_train), images_test, one_hot(labels_test)

def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

## == Neural Network == ##
# We will use a simple 3 layer feedforward neural network with one hidden layer
# Input layer: 784 neurons (28x28 pixels)
# Hidden layer: 10 neurons, ReLU activation
# Output layer: 10 neurons, Softmax activation

class SimpleNN:
    def __init__(self, hidden_layer_size=10, learning_rate=0.01, kaiming=True, use_mse=True):
        if not kaiming:
            self.W1 = np.random.rand(10, 784) - 0.5
            self.b1 = np.random.rand(10, 1) - 0.5
            self.W2 = np.random.rand(10, 10) - 0.5
            self.b2 = np.random.rand(10, 1) - 0.5
        else:
            self.W1 = np.random.randn(hidden_layer_size, 784) * np.sqrt(2 / 784)
            self.W2 = np.random.randn(10, hidden_layer_size) * np.sqrt(2 / hidden_layer_size)
            self.b1 = np.zeros((hidden_layer_size,1))
            self.b2 = np.zeros((10,1))

        self.learning_rate = learning_rate
        self.use_mse = use_mse
    
    def forward_prop(self, x):
        """
        A0 (784xM)  =   input layer
        Z1 (10xM)   =   unactivated hidden layer  = W1 (10x784) * A0 (784xM) + b1 (10x1)
        A1 (10xM)   =   activated hidden layer    = ReLU(Z1) 
        
        Z2 (10xM)   =   unactivated output layer  = W2 (10x10) * A1 (10xM) + b2 (10x1)
        A2 (10xM)   =   activated output layer    = softmax(Z2)
        """
        A0 = x
        self.Z1 = self.W1.dot(A0) + self.b1
        self.A1 = self.ReLU(self.Z1)
        
        Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = self.softmax(Z2)


    def backward_prop(self, x, y):
        """
        dZ2 (10xM)      = error = A2 (10xM) - Y (10xM)
        dW2 (10x10)     =       = (1/M) * dZ2 (10xM) * A1T (Mx10)
        db2 (10x1)      =       = (1/M) * SUM(dZ2 (10xM))

        dZ1 (10xM)      =       = W2T (10x10) * dZ2 (10xM) * ReLU_deriv(Z1 (10xM)) --> undo activation function using ReLU_deriv
        dW1 (10x784)    =       = (1/M) * dZ1 (10xM) * XT (10x784)
        db1 (10x1)      =       = (1/M) * SUM(dZ1 (10x1))
        """
        m = len(y[0])
        dZ2 = self.A2 - y
        dW2 = (1/m) * dZ2.dot(self.A1.T)
        db2 = (1/m) * dZ2.sum(axis=1, keepdims=True)
        
        dZ1 = self.W2.T.dot(dZ2) * self.ReLU_deriv(self.Z1)
        dW1 = (1/m) * dZ1.dot(x.T)
        db1 = (1/m) * dZ1.sum(axis=1, keepdims=True)
        return dW1, db1, dW2, db2


    def update_params(self, dW1, db1, dW2, db2):
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2

    def ReLU(self, x):
        return np.maximum(x, 0)
    
    def ReLU_deriv(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        x = x - np.max(x, axis=0, keepdims=True) 
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=0, keepdims=True)
    
    def compute_loss(self, Y):
        if self.use_mse:
            return (1/(2*len(Y[0]))) * np.sum((self.A2 - Y)**2)
        eps = 1e-9
        return -np.mean(np.sum(Y * np.log(self.A2 + eps), axis=0))
    
    def get_predictions(self):
        return np.argmax(self.A2, 0)
    
    def make_predictions(self, x):
        self.forward_prop(x)
        preds = self.get_predictions()
        return preds
    
    def save(self, path, accuracy_train, accuracy_test):
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            acc_train=accuracy_train,
            acc_test=accuracy_test
        )

    def load(self, path):
        data = np.load(path)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
        self.acc_train = data["acc_train"]
        self.acc_test = data["acc_test"]

def get_accuracy(predictions, Y):
    print(predictions, Y)
    Y = np.argmax(Y, axis=0)
    return np.mean(predictions == Y)

def main():
    file_name = './simple-nn.npz'
    images_train, labels_train, images_test, labels_test = load_data()
    
    if os.path.isfile(file_name):
        nn = SimpleNN()
        nn.load(file_name)
        print("Loaded saved model.")
        print("Train accuracy: ", nn.acc_train)
        print("Test accuracy: ", nn.acc_test)
    else:
        n_iterations = 800
        nn  = SimpleNN(hidden_layer_size=64, learning_rate=0.05)
        for i in range(n_iterations):
            nn.forward_prop(images_train)
            dW1, db1, dW2, db2 = nn.backward_prop(images_train, labels_train)
            nn.update_params(dW1, db1, dW2, db2)
            if i % 5 == 0:
                print("Iteration: ", i)
                loss = nn.compute_loss(labels_train)
                print(f"iter {i}, loss {loss:.4f}")
                preds = nn.get_predictions()
                accuracy_train = get_accuracy(preds, labels_train)
                print(get_accuracy(preds, labels_train))
        
        predictions_test = nn.make_predictions(images_test)
        accuracy_test = get_accuracy(predictions_test, labels_test)
        nn.save(file_name, accuracy_train, accuracy_test)
    
    from step_predict import run_step_predictor
    run_step_predictor(nn, images_test, labels_test)


if __name__ == "__main__":
    main()