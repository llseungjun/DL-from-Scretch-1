import numpy as np
from function import function_1, function_2, function_tmp1, function_tmp2

def numerical_diff(f, x):
    """
        basic code:
            h = 10e-50
            return (f(x+h) - f(x))/h
        위 코드처럼 하면 안되는 이유 2가지:
            1. h가 너무 작다
                - rounding error가 발생한다.
                - rounding error 관련 참고(https://ybeaning.tistory.com/35)
            2. (f(x+h) - f(x))/h 수식 자체의 오류
                - 정확하게 x 지점에서의 미분값(접선의 기울기)와는 차이를 보인다.
                - 수치 미분적으로 정확하게는 (f(x+h) - f(x-h))/2h가 더 오차가 작다.
                - 이를 중앙 차분이라 하고, 원래의 방식은 전방 차분이라고 한다.
                - 전방 차분, 중앙 차분 관련 참고(https://nanunzoey.tistory.com/entry/수치-미분Numerical-differentiation이란)
    """
    h = 1e-4 # 0.0001, 실험적으로 적당한 수치라고 알려짐(책 122쪽 참고)
    return (f(x+h) - f(x-h)) / (2*h)

# def numerical_gradient(f,x):
#     """
#         설명:
#             다차원 변수인 x에 대한 편미분 값을 한번에 구하는 함수
#         Arguments:
#             f -- 목적함수, def로 정의된 함수
#             x -- 편미분 하고자 하는 포인트, numpy array
#         return:
#             grad -- 목적함수 f의 각 지점 x에서의 편미분 값
#     """
#     h = 1e-4 # 0.0001
#     grad = np.zeros_like(x) # x와 형상이 같은 배열 생성

#     for idx in range(x.size): # 해당하는 i번째 x에만 수치 미분을 적용하기 위해 for문을 적용
#         tmp_val = x[idx]
#         # f(x+h) 계산
#         x[idx] = tmp_val + h # i번째 x에만 h를 더함, 나머지는 전부 원래의 값
#         fxh1 = f(x)

#         # f(x-h) 계산
#         x[idx] = tmp_val - h
#         fxh2 = f(x)
        
#         # 수치 미분 값 저장
#         grad[idx] = (fxh1 - fxh2) / (2*h)
        
#         # i번째 x값 원래대로 복원
#         x[idx] = tmp_val
    
#     return grad

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """함수 f위 최초의 지점 init_x에서 부터 극솟값을 찾기 위한 함수 
       초기 x 지점부터 수치 미분 값을 계속 구해서 새로운 지점 x로 이동한다.
    Args:
        f (func):
        init_x (ndarray): 최적화하기 위한 초기 x값
        lr (float): 학습률. Defaults to 0.01.
        step_num (int): 학습 횟수. Defaults to 100.
    """
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x 

if __name__ == "__main__":
    print(f" function_1의 5에서의 수치 미분 값 : {numerical_diff(function_1,5)} ")
    print(f" function_1의 10에서의 수치 미분 값 : {numerical_diff(function_1,10)} ")
    print("\n"+"*"*15 + " 정답은 책 125쪽에 있습니다 " + "*"*15+"\n")
    print(f" function_tmp의 x0 = 3, x1 = 4에서의 x0에 대한 수치 미분 값 : {numerical_diff(function_tmp1,3.0)} ")
    print(f" function_tmp의 x0 = 3, x1 = 4에서의 x1에 대한 수치 미분 값 : {numerical_diff(function_tmp2,4.0)} ")
    print("\n"+"*"*15 + " 정답은 책 126쪽에 있습니다 " + "*"*15+"\n")
    print(f" function_2의 입력 {np.array([3.0, 4.0])}에 대한 수치 미분 값 : {numerical_gradient(function_2,np.array([3.0, 4.0]))} ")
    print(f" function_2의 입력 {np.array([0.0, 2.0])}에 대한 수치 미분 값 : {numerical_gradient(function_2,np.array([0.0, 2.0]))} ")
    print(f" function_2의 입력 {np.array([3.0, 0.0])}에 대한 수치 미분 값 : {numerical_gradient(function_2,np.array([3.0, 0.0]))} ")
    print("\n"+"*"*15 + " 정답은 책 128쪽에 있습니다 " + "*"*15+"\n")
    print(f" function_2의 초기 지점 {np.array([-3.0, 4.0])}에서부터 100번 경사법을 적용한 후 값 \
          : {gradient_descent(function_2,init_x = np.array([-3.0, 4.0]),lr=0.1,step_num=100)} ")
    print("\n"+"*"*15 + " 정답은 책 132쪽에 있습니다 " + "*"*15+"\n")