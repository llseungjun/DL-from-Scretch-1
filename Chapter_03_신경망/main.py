import numpy as np
from network import network, mnist_network
import argparse

def main(args):
    if args.network == "3_layers_basic":
        """
            1. 새로운 신경망 인스턴스 생성
            2. 임의의 numpy array x 생성 후 신경망에 입력
            3. 신경망 순전파 결과 출력 후 종료
        """
        new_network = network() 
        x = np.array([1.0, 0.5])
        y = new_network.forward(x)
        print("순전파 결과 : ", y)

    elif args.network == "mnist":
        """
            1. 새로운 신경망 인스턴스 생성
            2. 임의의 numpy array x 생성 후 신경망에 입력
            3. 신경망 순전파 결과 출력 후 종료
        """
        new_network = mnist_network(args.batch)
        new_network.init_network() 
        new_network.get_accuracy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--network", "-n", default="3_layers_basic", type=str, choices=['3_layers_basic','mnist'] ,help='원하는 네트워크 이름 입력')
    parser.add_argument("--batch", "-b", default=False, type=bool,help='batch 학습을 적용할지 여부 T/F 선택')
    args = parser.parse_args()
    
    main(args)