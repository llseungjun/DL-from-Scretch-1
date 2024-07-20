import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from utils import sigmoid, identity_function, softmax
import pickle

class network:
    def __init__(self):
        """
            임의의 3층 신경망 설계
            Arguments:
                self.network -- dictionary of network parameters('W1','W2','W3,'B1','B2','B3')
            Returns:
                -
        """
        self.network = {}
        self.network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # (2, 3) , input x shape: (1,2) -> expectation output shape: (1, 3) 
        self.network['b1'] = np.array([0.1, 0.2, 0.3]) # (1, 3)
        self.network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # (3, 2), a2 shape: (1,3) -> expectation output shape: (1,2)
        self.network['b2'] = np.array([0.1, 0.2]) # (1, 2)
        self.network['W3'] = np.array([[0.1, 0.3],[0.2, 0.4]]) # (2, 2), a3 shape: (1,2) - > expectation output shape: (1,2)
        self.network['b3'] = np.array([0.1, 0.2]) # (1, 2)
    
    def forward(self, x):
        """
            입력 x에 대한 신경망의 순전파
            Arguments:
                x -- numpy array of shape (1, 2)
            Returns:
                y -- numpy array of shape (1, 2)
        """
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = identity_function(a3)
        
        return y

class mnist_network:
    def __init__(self, batch = False):
        """
            mnist 데이터 import
            Arguments:
                self.x_train -- x train of flatten image, shape(60000, 784)
                self.t_train -- label train of image, shape(60000, )
                self.x_test -- x test of flatten image, shape(10000, 784)
                self.t_test -- label test of image, shape(10000, )
        """
        (self.x_train, self.t_train), (self.x_test, self.t_test) = \
            load_mnist(flatten=True, normalize=False)
        self.batch = batch

    def init_network(self):
        """
            미리 저장된 가중치 pkl 파일 불러오기
            Arguments:
                self.network -- dictionary of pretrained weights & biases
        """
        with open("sample_weight.pkl",'rb') as f:
            network = pickle.load(f)
        self.network = network

    def predict(self, x):
        """
            주의사항:
                네트워크 파라미터 어딘가에 
                sigmoid에서 오버플로우 오류(아마 큰 음수를 만들어 내는)가 
                생기게 하는 부분이 있음
                
            추후 로깅 후 수정 예정
        """
        W1, W2, W3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)
        return y
    
    def get_accuracy(self):
        accuracy_cnt = 0
        if self.batch:
            batch_size = 100
            for i in range(0,len(self.x_test),batch_size):
                x_batch = self.x_test[i:i+batch_size]
                y_batch = self.predict(x_batch)
                p = np.argmax(y_batch, axis = 1)
                accuracy_cnt += np.sum(p == self.t_test[i:i+batch_size])
            print("batch size 100 test Accuracy:" + str(float(accuracy_cnt) / len(self.x_test)))
        else:
            for i in range(len(self.x_test)):
                y = self.predict(self.x_test[i])
                p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스 반환
                if p == self.t_test[i]:
                    accuracy_cnt += 1
            print("Accuracy:" + str(float(accuracy_cnt) / len(self.x_test)))
