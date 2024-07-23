import numpy as np
from convnet import SimpleConvNet
import argparse
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.optimizer import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

def main(args):
    # data import
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True,flatten=False)

    # import Neural net
    network = SimpleConvNet() # arguments는 default로 설정

    # optimizer 지정
    optimizer = Adam(lr = 0.01)

    if args.mode == 'train':
        iters_num  = 1000
        train_size = x_train.shape[0]
        batch_size = 100
        learning_rate = 0.01

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []
        
        iter_per_epoch = max(train_size/batch_size,1)

        pbar = tqdm(range(iters_num))
        for i in pbar:
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]
            
            # gradient 구하기
            grads = network.gradient(x_batch, t_batch)
            # 가중치 업데이트 'Adam'으로 업데이트
            optimizer.update(network.params,grads)

            # loss 구하기
            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                test_acc = network.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
        
        pbar.close()

        # 그래프 그리기
        markers = {'train': 'o', 'test': 's'}
        x = np.arange(iters_num / iter_per_epoch)
        plt.plot(x, train_acc_list, marker='o', label='train', markevery=2)
        plt.plot(x, test_acc_list, marker='s', label='test', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

    else:
        file_name="params.pkl"
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        # network의 기본 params 뿐만 아니라
        # 각 layer의 parameter들 또한 pickle에서 읽어온 parameter를 세팅해줘야 한다.
        for key, val in params.items():
            network.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            network.layers[key].W = network.params['W' + str(i+1)]
            network.layers[key].b = network.params['b' + str(i+1)]

        if args.mode == 'test':
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            print("train_acc, test_acc | " + str(train_acc) + ',' + str(test_acc))
        
        else:
            # reference : https://github.com/WegraLee/deep-learning-from-scratch/blob/master/ch07/visualize_filter.py
            FN, C, FH, FW = network.params['W1'].shape
            ny = int(np.ceil(FN / 8))

            fig = plt.figure()
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

            for i in range(FN):
                ax = fig.add_subplot(ny, 8, i+1, xticks=[], yticks=[])
                ax.imshow(network.params[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
            plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', '-m', default = 'train', choices = ['train','test','visualize_filter'])

    args = parser.parse_args()

    main(args)

