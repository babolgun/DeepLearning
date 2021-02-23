import numpy as np
from time import time
import gzip
from functions import one_hot_encoding
from functions import neural_network
from functions import sgd
from functions import fit
from functions import validate_accuracy


def main():
    start_time = time()
    print("---------- main3 --------------")
    f0 = gzip.open('/home/luca/data/mnist/train-images-idx3-ubyte.gz', 'r')
    f1 = gzip.open('/home/luca/data/mnist/t10k-images-idx3-ubyte.gz', 'r')
    l0 = gzip.open('/home/luca/data/mnist/train-labels-idx1-ubyte.gz', 'r')
    l1 = gzip.open('/home/luca/data/mnist/t10k-labels-idx1-ubyte.gz', 'r')
    X_train = np.frombuffer(f0.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    X_test = np.frombuffer(f1.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    y_train = np.frombuffer(l0.read(), dtype=np.uint8, offset=8)
    y_test = np.frombuffer(l1.read(), dtype=np.uint8, offset=8)

    y_train = one_hot_encoding(y_train)
    y_label = one_hot_encoding(y_test)
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train, X_test = X_train - mean, X_test - mean
    X_train, X_test = X_train / std, X_test / std

    model = neural_network((89, 'TanH'), (10, 'Softmax'), input_nodes=784, seed=20190119)
    model = fit(x_train=X_train, y_train=y_train, x_test=X_test, y_test=y_label,
                model=model,
                optimizer=sgd(epochs=50,
                              eta=0.15,
                              etaN=0.05,
                              decay_type='exponential',
                              beta=0.9),
                batch_size=60,
                eval_every=5,
                early_stop=True,
                loss_function='cross-entropy',
                seed=20190119)

    validate_accuracy(x_test=X_test, y_test=y_test, model=model)

    print("--- %s seconds ---" % (time() - start_time))


if __name__ == '__main__':
    main()
    """ 
    Adding momentum, the model validation accuracy is: 95.51%, (beta: 0.76,  lr: 0.4-0.2 exp);
                     the model validation accuracy is: 95.64%, (beta: 0.85, lr: 0.35-0.05 exp)
                     he model validation accuracy is: 95.17%   (beta: 0.9, lr: 0.15-0.05 exp)
    """
