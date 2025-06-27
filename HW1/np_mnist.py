import numpy as np
from tqdm import tqdm

X_train = np.load('./mnist/X_train.npy') # (60000, 784), 数值在0.0~1.0之间
y_train = np.load('./mnist/y_train.npy') # (60000, )
y_train = np.eye(10)[y_train] # (60000, 10), one-hot编码

X_val = np.load('./mnist/X_val.npy') # (10000, 784), 数值在0.0~1.0之间
y_val = np.load('./mnist/y_val.npy') # (10000,)
y_val = np.eye(10)[y_val] # (10000, 10), one-hot编码

X_test = np.load('./mnist/X_test.npy') # (10000, 784), 数值在0.0~1.0之间
y_test = np.load('./mnist/y_test.npy') # (10000,)
y_test = np.eye(10)[y_test] # (10000, 10), one-hot编码

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1. - sigmoid(x))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def loss_fn(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def loss_fn_prime(y_true, y_pred):
    return y_pred - y_true

def init_weights(shape=()):
    return np.random.normal(loc=0.0, scale=np.sqrt(2.0/shape[0]), size=shape)

class Network(object):
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):

        self.W1 = init_weights((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = init_weights((hidden_size, hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = init_weights((hidden_size, output_size))
        self.b3 = np.zeros((1, output_size))
        self.lr = lr

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = softmax(self.z3)
        return self.a3

    def step(self, x_batch, y_batch):

        y_pred = self.forward(x_batch)
        
        loss = loss_fn(y_batch, y_pred)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_batch, axis=1))
        
        dL_dz3 = loss_fn_prime(y_batch, y_pred)
        dL_dW3 = np.dot(self.a2.T, dL_dz3)
        dL_db3 = np.sum(dL_dz3, axis=0, keepdims=True)

        dL_dz2 = np.dot(dL_dz3, self.W3.T) * (self.a2 * (1 - self.a2))
        dL_dW2 = np.dot(self.a1.T, dL_dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)
        
        dL_dz1 = np.dot(dL_dz2, self.W2.T) * relu_prime(self.z1)
        dL_dW1 = np.dot(x_batch.T, dL_dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)
        
        self.W3 -= self.lr * dL_dW3
        self.b3 -= self.lr * dL_db3 
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2       
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1
        
        return loss, accuracy

if __name__ == '__main__':

    net = Network(input_size=784, hidden_size=356, output_size=10, lr=0.015)
    for epoch in range(10):
        losses = []
        accuracies = []
        p_bar = tqdm(range(0, len(X_train), 64))
        for i in p_bar:
            x_batch = X_train[i:i+64]
            y_batch = y_train[i:i+64]
            loss, accuracy = net.step(x_batch, y_batch)
            losses.append(loss)
            accuracies.append(accuracy)
            p_bar.set_description(f"Epoch {epoch+1} Loss: {np.mean(losses):.4f} Accuracy: {np.mean(accuracies):.4f}")
        y_val_pred = net.forward(X_val)
        val_loss = loss_fn(y_val, y_val_pred)
        val_accuracy = np.mean(np.argmax(y_val_pred, axis=1) == np.argmax(y_val, axis=1))
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f} Val Accuracy: {val_accuracy:.4f}")
    y_test_pred = net.forward(X_test)
    test_accuracy = np.mean(np.argmax(y_test_pred, axis=1) == np.argmax(y_test, axis=1))
    print("Test Accuracy: {:.4f}".format(test_accuracy))