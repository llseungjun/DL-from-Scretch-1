import numpy as np
from layers import AddLayer, MulLayer
import argparse

def main(args):
    if args.buy == 'apple':
        # buy_apple(162p) 예시 코드
        apple = 100
        apple_num = 2
        tax = 1.1

        mul_apple_layer = MulLayer()
        mul_tax_layer = MulLayer()

        apple_price = mul_apple_layer.forward(apple,apple_num)
        price = mul_tax_layer.forward(apple_price,tax)

        dprice = 1
        dapple_price, dtax = mul_tax_layer.backward(dprice)
        dapple, dapple_num = mul_apple_layer.backward(dapple_price)

        print("d사과, d사과의 개수, d소비세 | " + str(dapple) + ',' + str(int(dapple_num)) + ',' + str(dtax))
        ###
    else:
        apple = 100
        apple_num = 2
        orange = 150
        orange_num = 3
        tax = 1.1

        mul_apple_layer = MulLayer()
        mul_orange_layer = MulLayer()
        add_apple_orange_layer = AddLayer()
        mul_tax_layer = MulLayer()

        apple_price = mul_apple_layer.forward(apple, apple_num)
        orange_price = mul_orange_layer.forward(orange, orange_num)
        all_price = add_apple_orange_layer.forward(apple_price,orange_price)
        price = mul_tax_layer.forward(all_price, tax)

        dprice = 1
        dall_price, dtax = mul_tax_layer.backward(dprice)
        dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
        dorange, dorange_num = mul_orange_layer.backward(dorange_price)
        dapple, dapple_num = mul_apple_layer.backward(dapple_price)

        print(price)
        print(int(dapple_num), dapple, round(dorange,1), int(dorange_num), dtax) # 110, 2.2, 3.3, 165, 650 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--buy','-b',default='apple',choices=['apple','apple_orange'])

    args = parser.parse_args()

    main(args)