import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step_function(x):
    """
        if x > 0:
            return 1
        else:
            return 0
       
        위와 같은 구현도 가능하지만, 
        x가 numpy array일 경우 error가 발생.
        따라서 아래와 같이 구현해야 numpy array일 경우에도 정상 작동
    
    """
    y = x > 0
    return y.astype(np.int)

def relu(x):
    return np.maximum(x,0)

def identity_function(x):
    """
        출력층의 activation function
        항등함수로 구현했다.
    """
    return x

def softmax(a):
    """
        exp_a = np.exp(a)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        위와 같이 구현해도 되지만, 
        a값이 너무 커질 경우 지수함수인 np.exp의 값이 무한대로 커져버려서 오버플로우가 생길 수 있다.
        따라서 입력 a에 적당한 값 c를 빼주어 아래와 같이 구현해야 에러를 해결할 수 있다. 
    """
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y