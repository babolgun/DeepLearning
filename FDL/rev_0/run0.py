import numpy as np
from numpy import ndarray
from time import time
from typing import List
from typing import Tuple
from copy import deepcopy
import gzip
import matplotlib.pyplot as plt


def add_layer(model: List, input_nodes: int, output_nodes: int, activation: str, seed: int = 1) -> List:
    w = init_weights(output_nodes, input_nodes, seed=seed)
    b = init_biases(output_nodes, seed=seed)
    params = (w, b)
    layer = [params, activation]
    model.append(layer)
    return model


def d_phi_dz(x: ndarray, activation: str) -> ndarray:
    if activation == 'Identity':
        return np.ones_like(x)
    elif activation == 'TanH':
        return 1. - np.tanh(x) * np.tanh(x)
    elif activation == 'ReLU':
        return np.heaviside(x, np.ones_like(x))
    elif activation == 'Sigmoid':
        # x = np.array(x, dtype=np.float64)
        x = np.clip(x, -500, 500)
        sigma = 1./(1. + np.exp(-x))
        return sigma * (1. - sigma)
    else:
        return Exception


def forward(x0: ndarray, model: List):
    x = x0
    for layer in model:
        w = layer[0][0]
        b = layer[0][1]
        z = np.add(np.matmul(w, x), b)
        x = phi(x=z, activation=layer[1])
    return x


def forward_pass(x0: ndarray, model: List):
    xk = []
    d_phi = []
    x = x0
    for layer in model:
        w, b = layer[0]
        z = np.add(np.matmul(w, x), b)
        d_phi.append(d_phi_dz(x=z, activation=layer[1]))
        x = phi(x=z, activation=layer[1])
        xk.append(x)
    return xk, d_phi


def generate_batches(x: ndarray, y: ndarray, size: int = 1) -> Tuple[ndarray]:
    n = x.shape[1]
    for i in range(0, n, size):
        x_batches, y_batches = x[:, i: i+size], y[:, i: i+size]
        yield x_batches, y_batches


def init_biases(rowNum: int, scale: float = 1., seed: int = 1) -> ndarray:
    np.random.seed(seed)
    b = np.random.normal(loc=0., scale=scale, size=(rowNum, 1))
    return b


def init_weights(rowNum: int, colNum: int, scale: float = 1., seed: int = 1) -> ndarray:
    np.random.seed(seed)
    w = np.random.normal(loc=0., scale=scale, size=(rowNum, colNum))
    return w


def mse_loss(y: ndarray, yc: ndarray) -> float:
    loss = np.sum(np.power(yc - y, 2) / y.shape[1])
    return loss


def neural_network(*args: Tuple, input_nodes: int, seed: int) -> List:
    model = []
    for arg in args:
        add_layer(seed=seed, model=model, input_nodes=input_nodes, output_nodes=arg[0], activation=arg[1])
        input_nodes = arg[0]
    return model


def one_hot_encoding(y):
    num_labels = len(y)
    labels = np.zeros((num_labels, 10))
    for i in range(num_labels):
        labels[i][y[i]] = 1
    return labels


def permute_data(X, y):
    perm = np.random.permutation(X.shape[1])
    return X[:, perm], y[:, perm]


def phi(x: ndarray, activation: str) -> ndarray:
    if activation == 'Identity':
        return x
    elif activation == 'TanH':
        return np.tanh(x)
    elif activation == 'ReLU':
        return np.maximum(x, 0)
    elif activation == 'Sigmoid':
        x = np.array(x, dtype=np.float64)
        x = np.clip(x, -500, 500)
        return 1./(1. + np.exp(-x))
    else:
        return Exception


def train_batch(x0: ndarray, y: ndarray, model: List, eta: float) -> List:
    xk, d_phi = forward_pass(x0=x0, model=model)
    n = len(model)
    # print(n)
    eps = xk[n-1] - y
    eps = eps.mean(axis=1).reshape(y.shape[0], 1)
    dC_dx = 2./x0.shape[0]*eps
    for k in reversed(range(n)):

        if k == n - 1:
            dC_db = d_phi[k].mean(axis=1, keepdims=True) * dC_dx
        else:
            w1 = model[k + 1][0][0]
            dC_db = np.matmul(w1.T, dC_db)
            dC_db = d_phi[k].mean(axis=1, keepdims=True) * dC_db
        if k == 0:
            dx_dw = x0.mean(axis=1, keepdims=True)
        else:
            dx_dw = xk[k-1].mean(axis=1, keepdims=True)

        dC_dw = np.matmul(dC_db, dx_dw.T)
        (w, b) = model[k][0]
        b = b - eta * dC_db
        w = w - eta * dC_dw
        # print("dC_db: ", dC_db.shape)
        # print("dC_dw: ", dC_dw.shape)
        # print("end layer %d ---------------------" % k)
        model[k][0] = (w, b)

    return model

# def train_batch(x0: ndarray, y: ndarray, model: List, eta: float) -> List:
#     xk, d_phi = forward_pass(x0=x0, model=model)
#     n = len(model)
#     eps = xk[n - 1] - y
#     dC_dx = 2. / y.shape[0] * eps
#
#     for k in reversed(range(n)):
#
#         if k == n - 1:
#             dC_dx = dC_dx * d_phi[k]
#         else:
#             w1 = model[k + 1][0][0]
#             dC_dx = np.matmul(w1.T, dC_dx) * d_phi[k]
#
#         if k == 0:
#             dx_dw = x0
#         else:
#             dx_dw = xk[k - 1]
#
#         dC_dw = np.einsum('ik,jk->ijk', dC_dx, dx_dw)
#
#         (w, b) = model[k][0]
#         b = b - eta * dC_dx.mean(axis=1, keepdims=True)
#         w = w - eta * dC_dw.mean(axis=2, keepdims=True).squeeze()
#
#         model[k][0] = (w, b)
#
#     return model


def train(x_train: ndarray,
          y_train: ndarray,
          x_test: ndarray,
          y_test: ndarray,
          model: List,
          eta: float,
          epochs: int = 1,
          batch_size: int = 1,
          eval_every: int = 1) -> List:

    for e in range(epochs):

        batch_generator = generate_batches(x=x_train, y=y_train, size=batch_size)
        for i, (x_batch, y_batch) in enumerate(batch_generator):
            model = train_batch(x0=x_batch, y=y_batch, model=model, eta=eta)

        if (e+1) % eval_every == 0:
            test_predictions = forward(x0=x_test, model=model)
            loss = mse_loss(test_predictions, y_test)

            print()
            print(f"""Loss after epoch {e + 1} was {loss:.3f}, 
                                using the model from epoch {e + 1 - eval_every}""")

    return model


def validate_accuracy(x_test: ndarray, y_test: ndarray, model: List) -> None:
    predictions = forward(x_test, model=model)
    accuracy = np.equal(np.argmax(predictions, axis=0), y_test).sum() * 100 / y_test.shape[0]
    return print(f'''The model validation accuracy is: {accuracy:.6f}%''')


def main():
    start_time = time()
    f0 = gzip.open('/home/luca/data/mnist/train-images-idx3-ubyte.gz', 'r')
    f1 = gzip.open('/home/luca/data/mnist/t10k-images-idx3-ubyte.gz', 'r')
    l0 = gzip.open('/home/luca/data/mnist/train-labels-idx1-ubyte.gz', 'r')
    l1 = gzip.open('/home/luca/data/mnist/t10k-labels-idx1-ubyte.gz', 'r')
    X_train = np.frombuffer(f0.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    X_test = np.frombuffer(f1.read(), dtype=np.uint8, offset=16).reshape(-1, 28 * 28)
    y_train = np.frombuffer(l0.read(), dtype=np.uint8, offset=8)
    y_test = np.frombuffer(l1.read(), dtype=np.uint8, offset=8)
    print('x = -----------------------------------------')

    x_train = X_train.T
    y_train = one_hot_encoding(y_train).T
    x_label = X_test.T
    y_label = one_hot_encoding(y_test).T

    model = neural_network((89, 'TanH'), (10, 'Sigmoid'), input_nodes=784, seed=20190119)
    # yr = forward(x0=x[:, :1], model=model)
    # train_batch(x0=x[:, :5], y=y[:, :5], model=model, eta=0.1)
    # yr1 = forward(x0=x[:, :1], model=model)
    # print(yr1 - yr)

    model = train(x_train, y_train, x_label, y_label, model, eta=0.15, epochs=50, batch_size=60, eval_every=10)
    validate_accuracy(x_test=x_label, y_test=y_test.T, model=model)

    print("--- %s seconds ---" % (time() - start_time))


if __name__ == '__main__':
    main()
