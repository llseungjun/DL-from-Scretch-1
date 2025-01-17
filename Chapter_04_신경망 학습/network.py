import numpy as np
from function import cross_entropy_error, softmax, sigmoid
from utils import gradient_descent, numerical_gradient

class simpleNet:
    def __init__(self) -> None:
        self.W = np.random.randn(2,3) # 2X3의 정규분포로 초기화
    
    def predict(self,x) -> np.ndarray:
        return np.dot(x,self.W)

    def loss(self, x, t) -> float:
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss    

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std = 0.01):
        """2층 신경망 클래스 가중치 초기화

        Args:
            input_size (int): input data의 차원
            hidden_size (int): hidden node의 개수
            output_size (int): output data의 차원
            weight_init_std (float): . Defaults to 0.01.
        """
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self,x):

        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y
    
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x,t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads        