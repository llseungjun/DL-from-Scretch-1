import numpy as np
import sys,os
sys.path.append(os.pardir)
from network import TwoLayerNet
from dataset.mnist import load_mnist
import argparse

def main(args):
    if args.mode == 'batch_train':
        """미니배치로 2층 신경망 학습
           train data 만 사용해서 수치 미분 활용한 경사하강 이용해서 파라미터 학습하는 과정
        """
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

        train_loss_list = []

        iters_num = 10000
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = 0.1

        network = TwoLayerNet(input_size = 784, hidden_size=50,output_size=10)

        for i in range(iters_num):
            # 미니배치
            batch_mask = np.random.choice(train_size,batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 기울기 계산
            grad = network.numerical_gradient(x_batch,t_batch)

            # 매개변수 갱신
            for key in ('W1','b1','W2','b2'):
                network.params[key] -= learning_rate * grad[key]
            
            # 학습 결과 기록
            loss = network.loss(x_batch,t_batch)
            train_loss_list.append(loss)
        
        print(f"최종 학습 결과 loss : {train_loss_list[-1]}")
    
    else:
        (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

        network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

        iters_num = 10000
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = 0.1

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        # 1 epoch당 반복 수
        iter_per_epoch = max(train_size / batch_size, 1)

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 기울기 계산
            grad = network.numerical_gradient(x_batch,t_batch)
            
            # 매개변수 갱신
            for key in ('W1','b1','W2','b2'):
                network.params[key] -= learning_rate * grad[key]
            
            # 학습 결과 기록
            loss = network.loss(x_batch,t_batch)
            train_loss_list.append(loss)

            # 1 epoch 당 정확도 계산
            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train,t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("train acc, test acc |" + str(train_acc) + ',', str(test_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mode', '-m', default='batch_train',choices=['batch_train', 'get_acc'])

    args = parser.parse_args()

    main(args)