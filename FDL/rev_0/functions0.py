import numpy as np
from copy import deepcopy
from numpy import ndarray
from typing import List
from typing import Tuple


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


def fit(x_train: ndarray,
        y_train: ndarray,
        x_test: ndarray,
        y_test: ndarray,
        model: List,
        optimizer: List,
        epochs: int = 1,
        batch_size: int = 1,
        eval_every: int = 1,
        early_stop: bool = False) -> List:

    best_loss = 1e9

    for e in range(epochs):

        if (e+1) % eval_every == 0:
            last_model = deepcopy(model)

        x_train, y_train = permute_data(x_train, y_train)
        batch_generator = generate_batches(x=x_train, y=y_train, size=batch_size)
        eta = optimizer[e]

        for i, (x_batch, y_batch) in enumerate(batch_generator):
            model = train_batch(x0=x_batch, y=y_batch, model=model, eta=eta)

        if (e+1) % eval_every == 0:
            test_predictions = forward(x0=x_test, model=model)
            loss = sqr_mse(test_predictions, y_test)
            if early_stop:
                if loss < best_loss:
                    print(f"Validation loss after {e + 1} epochs is {loss:.3f}")
                    best_loss = loss
                else:
                    print()
                    print(f"""Loss increased after epoch {e + 1}, final loss was {best_loss:.3f}, 
                                using the model from epoch {e + 1 - eval_every}""")
                    model = last_model
                    break

    return last_model


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


def forward(x0: ndarray, model: List):
    x = x0
    for layer in model:
        w = layer[0][0]
        b = layer[0][1]
        z = np.add(np.matmul(w, x), b)
        x = phi(x=z, activation=layer[1])
    return x


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


def sqr_mse(y: ndarray, xL: ndarray) -> float:
    loss = np.sum(np.power(xL-y, 2)) / y.shape[1]
    return np.sqrt(loss)


def neural_network(*args: Tuple, input_nodes: int, seed: int) -> List:
    model = []
    for arg in args:
        add_layer(seed=seed,
                  model=model,
                  input_nodes=input_nodes,
                  output_nodes=arg[0],
                  activation=arg[1])
        input_nodes = arg[0]
    return model


def one_hot_encoding(y):
    num_labels = len(y)
    labels = np.zeros((num_labels, 10))
    for i in range(num_labels):
        labels[i][y[i]] = 1
    return labels


def optimize(epochs: int, eta0: float, etaN: float, decay_type: str = 'linear') -> List:
    etas = []
    eta = eta0
    if decay_type == 'linear':
        step = (eta0 - etaN) / (epochs - 1)
        for i in range(epochs):
            eta = eta - step
            etas.append(eta)
    elif decay_type == 'none':
        for i in range(epochs):
            etas.append(eta0)
    return etas


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
    eps = xk[n - 1] - y
    dC_dx = 2. / y.shape[1] * eps.mean(axis=1, keepdims=True)
    # print('dC_dx:', dC_dx.shape)

    for k in reversed(range(n)):

        if k == n - 1:
            dC_dx = (dC_dx * d_phi[k]).mean(axis=1, keepdims=True)
            # print('dC_dx:', dC_dx.shape)
        else:
            w1 = model[k + 1][0][0]
            dC_dx = (np.matmul(w1.T, dC_dx) * d_phi[k]).mean(axis=1, keepdims=True)
            # print('dC_dx:', dC_dx.shape)

        if k == 0:
            dx_dw = x0.mean(axis=1, keepdims=True)
            # print('dx_dw:', dx_dw.shape)
        else:
            dx_dw = xk[k - 1].mean(axis=1, keepdims=True)
            # print('dx_dw:', dx_dw.shape)

        dC_dw = np.matmul(dC_dx, dx_dw.T)

        (w, b) = model[k][0]
        # note that dx_db = 1 => dC_db = dC_dx
        b = b - eta * dC_dx
        w = w - eta * dC_dw

        model[k][0] = (w, b)

    return model


def validate_accuracy(x_test: ndarray, y_test: ndarray, model: List) -> None:
    predictions = forward(x_test, model=model)
    accuracy = np.equal(np.argmax(predictions, axis=0), y_test).sum() * 100 / y_test.shape[0]
    return print(f'''The model validation accuracy is: {accuracy:.2f}%''')
