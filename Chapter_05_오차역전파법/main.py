import numpy as np
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from layers import TwoLayerNet
import argparse

def main(args):
    
    # data import 
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # network import
    network = TwoLayerNet(input_size=784, hidden_size=50,output_size=10)
    
    if args.mode == 'gradient_check':
        x_batch = x_train[:3]
        t_batch = t_train[:3]

        grad_numrical = network.numerical_gradient(x_batch,t_batch)
        grad_backprop = network.gradient(x_batch,t_batch)

        for key in grad_numrical.keys():
            diff = np.average(np.abs(grad_backprop[key] -  grad_numrical[key]))
            print(key + ":" + str(diff))
    else:
        iters_num = 10000
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = 0.1

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        iter_per_epoch = max(train_size / batch_size ,1)

        for i in range(iters_num):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 기울기 계산
            grads = network.gradient(x_batch,t_batch)

            # 파라미터 갱신
            for key in ['W1','b1','W2','b2']:
                network.params[key] -= learning_rate * grads[key]
            
            loss = network.loss(x_batch,t_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = network.accuaracy(x_train, t_train)
                test_acc = network.accuaracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("train acc, test acc |" + str(train_acc) + ',', str(test_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode','-m',default='train',choices=['train','gradient_check'])

    args = parser.parse_args()

    main(args)
    

