import numpy as np

def mean_squared_error(y, t):
    """
        Q. y,t의 차원 만큼으로 나눠야 할거 같은데 왜 0.5를 곱한걸까..?
    """
    return 0.5*np.sum(np.sqrt((y-t)**2))

def cross_entropy_error(y, t):
    """
        basic code:
            delta = 1e-7 # inf 방지
            return -np.sum(t*np.log(y+delta))    
        설명:
            데이터를 하나씩 처리할 때는 위와 같이 코드를 작성할 수 있다.
            반면 미니배치를 위한 코드는 아래와 같다.
    """
    delta = 1e-7
    if y.ndim == 1: 
        # 데이터 하나씩 처리할 경우
        # 아래 batch_size를 얻을 때 오류를 피하기 위해 reshape을 진행
        t = t.reshape(1,t.size)
        y = y.reshape(1,y.size)
    
    batch_size = y.shape[0]

    if t.size == 1: 
        # 정답 레이블이 원-핫 인코딩일 경우
        # 약간의 기교 같은데, 원리를 까먹은거 같을 땐 책 118-119쪽 참고
        return -np.sum(np.log(y[np.arrange(batch_size),t] +  delta)) / batch_size
    else:
        return -np.sum(t*np.log(y+delta)) / batch_size

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def sigmoid(x):
    return 1/(1+np.exp(-x))

def function_1(x): # 수치 미분 예시용 함수
    return 0.01*x**2 + 0.1*x

def function_2(x):
    return x[0]**2 + x[1]**2

def function_tmp1(x0):
    return x0*x0 + 4.0**2.0

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1