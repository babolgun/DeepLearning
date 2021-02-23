import numpy as np
from time import time
import gzip
from functions0 import one_hot_encoding
from functions0 import neural_network
from functions0 import fit
from functions0 import validate_accuracy


def main():
    start_time = time()
    print("---------- main1 --------------")
    f0 = gzip.open('/home/luca/data/mnist/train-images-idx3-ubyte.gz', 'r')
    f1 = gzip.open('/home/luca/data/mnist/t10k-images-idx3-ubyte.gz', 'r')
    l0 = gzip.open('/home/luca/data/mnist/train-labels-idx1-ubyte.gz', 'r')
    l1 = gzip.open('/home/luca/data/mnist/t10k-labels-idx1-ubyte.gz', 'r')
    X_train = np.frombuffer(f0.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    X_test = np.frombuffer(f1.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    y_train = np.frombuffer(l0.read(), dtype=np.uint8, offset=8)
    y_test = np.frombuffer(l1.read(), dtype=np.uint8, offset=8)

    x_train = X_train.T
    y_train = one_hot_encoding(y_train).T
    x_label = X_test.T
    y_label = one_hot_encoding(y_test).T

    model = neural_network((89, 'TanH'), (10, 'Sigmoid'), input_nodes=784, seed=20190119)
    print(model[0][0][0].shape)
    print(model[0][0][1].shape)
    print(model[1][0][0].shape)
    print(model[1][0][1].shape)

    # model = fit(x_train=x_train,
    #             y_train=y_train,
    #             x_test=x_label,
    #             y_test=y_label,
    #             model=model,
    #             eta=.5,
    #             epochs=50,
    #             batch_size=60,
    #             eval_every=5,
    #             early_stop=True)

    validate_accuracy(x_test=x_label, y_test=y_test.T, model=model)

    print()
    print("--- %s seconds ---" % (time() - start_time))


if __name__ == '__main__':
    main()
    """ After experimenting different solutions on the backpropagation on main0 which resulted in inefficient code,
        I have come to the final implementation that still gives me some doubts due to different values in the
        parameters where I was expecting something closer to the Object Oriented solution offered by S. Weidman DLFS 
        The result  in terms of accuracy though, is very similar:
        I obtained validation accuracy 53.17% with eta (i.e. learning rate, 'lr' in the OO implementation).
        Now let's see what happens after data normalisation
    """
