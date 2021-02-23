import numpy as np
from time import time
import gzip
from functions0 import one_hot_encoding
from functions0 import neural_network
from functions0 import fit
from functions0 import validate_accuracy
from functions0 import optimize


def main():
    start_time = time()
    print("--------- main2 -----------")
    f0 = gzip.open('/home/luca/data/mnist/train-images-idx3-ubyte.gz', 'r')
    f1 = gzip.open('/home/luca/data/mnist/t10k-images-idx3-ubyte.gz', 'r')
    l0 = gzip.open('/home/luca/data/mnist/train-labels-idx1-ubyte.gz', 'r')
    l1 = gzip.open('/home/luca/data/mnist/t10k-labels-idx1-ubyte.gz', 'r')
    X_train = np.frombuffer(f0.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    X_test = np.frombuffer(f1.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    y_train = np.frombuffer(l0.read(), dtype=np.uint8, offset=8)
    y_test = np.frombuffer(l1.read(), dtype=np.uint8, offset=8)

    # data normalisation with respect to the mean and std deviation of X_train
    mean = np.mean(X_train)
    std = np.std(X_train)
    X_train, X_test = X_train - mean, X_test - mean
    X_train, X_test = X_train / std, X_test / std

    x_train = X_train.T
    y_train = one_hot_encoding(y_train).T
    x_label = X_test.T
    y_label = one_hot_encoding(y_test).T

    model = neural_network((89, 'ReLU'), (10, 'Sigmoid'), input_nodes=784, seed=20190119)
    optimizer = optimize(epochs=50, eta0=40., etaN=10., decay_type='none')
    model = fit(x_train=x_train,
                y_train=y_train,
                x_test=x_label,
                y_test=y_label,
                model=model,
                optimizer=optimizer,
                epochs=50,
                batch_size=60,
                eval_every=5,
                early_stop=True)

    validate_accuracy(x_test=x_label, y_test=y_test.T, model=model)

    print()
    print("--- %s seconds ---" % (time() - start_time))


if __name__ == '__main__':
    main()
    """
       The model validation accuracy is: 64.87% Rather disappointing, with eta = 6.5
       let's tune the learning rate:
       The model validation accuracy is: 65.09% with eta = 6.6
       The model validation accuracy is: 65.37% with eta = 6.7
       The model validation accuracy is: 65.92% with eta = 7.0
       The model validation accuracy is: 68.15% with eta = 8.0
       The model validation accuracy is: 69.13% with eta = 9.0
       The model validation accuracy is: 70.07% with eta = 10.0
       The model validation accuracy is: 71.54% with eta = 11.0
       The model validation accuracy is: 71.71% with eta = 12.0
       The model validation accuracy is: 72.85% with eta = 13.0   -- wich brings us to the OO result
       The model validation accuracy is: 73.62% with eta = 14.0
       The model validation accuracy is: 74.05% with eta = 15.0
       The model validation accuracy is: 75.34% with eta = 16.0
       The model validation accuracy is: 75.82% with eta = 17.0
       The model validation accuracy is: 76.09% with eta = 18.0
       The model validation accuracy is: 76.35% with eta = 19.0
       The model validation accuracy is: 75.79% with eta = 20.0
       
       Surprisingly the optimal result is obtained with a very different value of the the learning rate (here 19.5)
       I think the reason lies on an averaging applied on the number of samples rather then the number of nodes as I 
       did. This means I have to rework on the train_batch function 
       
       Last attempt with all sums instead of averages and eta=0.75e-4 The model validation accuracy is: 79.86%
    """
