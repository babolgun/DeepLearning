import numpy as np
from copy import deepcopy
from numpy import ndarray
from typing import List
from typing import Tuple
from scipy.special import logsumexp


def add_layer(model: List, input_nodes: int, output_nodes: int, activation: str, seed: int = 1,
              weight_init: str = 'std') -> List:
    scale = 1.
    if weight_init == 'scaled':
        scale = 2./(input_nodes + output_nodes)
    params = init_params(input_nodes, output_nodes, seed=seed, scale=scale)
    param_grads = init_param_grads(input_nodes, output_nodes)
    layer = [params, activation, param_grads]
    model.append(layer)
    return model


def cross_entropy(x: ndarray, y: ndarray):
    eps = 1e-9
    # clipping x values to avoid instability
    x = np.clip(x, eps, 1. - eps)
    loss = -y * np.log(x) - (1.-y) * np.log(1. - x)
    return np.sum(loss) / x.shape[0]


def df_dz(x: ndarray, activation: str) -> ndarray:
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
    elif activation == 'Softmax':
        return np.ones_like(x)
    else:
        return Exception


def f(x: ndarray, activation: str) -> ndarray:
    if activation == 'Identity':
        return x
    elif activation == 'TanH':
        return np.tanh(x)
    elif activation == 'ReLU':
        return np.maximum(x, 0)
    elif activation == 'Sigmoid':
        x = np.clip(x, -500, 500)
        return 1./(1. + np.exp(-x))
    elif activation == 'Softmax':
        return softmax(x, axis=1)
    else:
        return Exception


def fit(x_train: ndarray,
        y_train: ndarray,
        x_test: ndarray,
        y_test: ndarray,
        model: List,
        optimizer: List,
        batch_size: int = 1,
        eval_every: int = 1,
        early_stop: bool = False,
        loss_function: str = 'mse',
        seed: int = 1,
        dropout: float = 1.) -> List:

    best_loss = 1e9
    last_model = model
    np.random.seed(seed)
    etas, beta = optimizer
    epochs = len(etas)

    for e in range(epochs):

        if (e + 1) % eval_every == 0:
            # for early stopping
            last_model = deepcopy(model)

        x_train, y_train = permute_data(x_train, y_train)
        batch_generator = generate_batches(x=x_train, y=y_train, size=batch_size)
        eta = etas[e]

        for i, (x_batch, y_batch) in enumerate(batch_generator):
            model = train_batch(x0=x_batch, y=y_batch, model=model, eta=eta, beta=beta,
                                loss_function=loss_function, dropout=dropout)

        if (e+1) % eval_every == 0:
            test_predictions = forward(x0=x_test, model=model)
            loss = cross_entropy(test_predictions, y_test)
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

    return model


def forward(x0: ndarray, model: List):
    x = x0
    for layer in model:
        w, b = layer[0]
        z = np.add(np.matmul(x, w), b)
        x = f(x=z, activation=layer[1])
    return x


def forward_pass(x0: ndarray, model: List):
    xk = []
    d_phi = []
    x = x0
    for layer in model:
        w, b = layer[0]
        z = np.add(np.matmul(x, w), b)
        d_phi.append(df_dz(x=z, activation=layer[1]))
        x = f(x=z, activation=layer[1])
        xk.append(x)
    return xk, d_phi


def generate_batches(x: ndarray, y: ndarray, size: int = 1) -> Tuple[ndarray]:
    n = x.shape[0]
    for i in range(0, n, size):
        x_batches, y_batches = x[i: i+size], y[i: i+size]
        yield x_batches, y_batches


def init_params(rowNum: int, colNum: int, scale: float = 1., seed: int = 1) -> Tuple:
    np.random.seed(seed)
    w = np.random.normal(loc=0., scale=scale, size=(rowNum, colNum))
    b = np.random.normal(loc=0., scale=scale, size=(1, colNum))
    return w, b


def init_param_grads(rowNum: int, colNum: int) -> Tuple:
    dw = np.zeros((rowNum, colNum))
    db = np.zeros((1, colNum))
    return dw, db


def mse(y: ndarray, x: ndarray) -> float:
    loss = np.sum(np.power(x - y, 2)) / y.shape[0]
    return loss


def neural_network(*args: Tuple, weight_init: str = 'std', input_nodes: int, seed: int) -> List:
    model = []
    for arg in args:
        add_layer(model=model,
                  input_nodes=input_nodes,
                  output_nodes=arg[0],
                  activation=arg[1],
                  seed=seed,
                  weight_init=weight_init
                  )
        input_nodes = arg[0]
    return model


def one_hot_encoding(y):
    num_labels = len(y)
    labels = np.zeros((num_labels, 10))
    for i in range(num_labels):
        labels[i][y[i]] = 1
    return labels


def permute_data(X, y):
    perm = np.random.permutation(X.shape[0])
    return X[perm], y[perm]


def sgd(epochs: int, eta: float, etaN: float = 0., decay_type: str = 'none', beta: float = 0.):
    etas = []
    lr = eta
    if decay_type == 'linear':
        step = (eta - etaN) / (epochs - 1)
        etas.append(lr)
        for i in range(epochs-1):
            lr -= step
            etas.append(lr)
    elif decay_type == 'exponential':
        alpha = np.power(etaN / eta, 1. / (epochs - 1))
        etas.append(lr)
        for i in range(epochs-1):
            lr *= alpha
            etas.append(lr)
    elif decay_type == 'exp-growth':
        alpha = np.power(eta / etaN, 1. / (epochs - 1))
        print("alpha: ", alpha)
        for i in range(epochs):
            lr *= alpha
            etas.append(lr)
    elif decay_type == 'none':
        for i in range(epochs):
            etas.append(lr)
    return etas, beta


def softmax(x, axis=None):
    return np.exp(x - logsumexp(x, axis=axis, keepdims=True))


def train_batch(x0: ndarray, y: ndarray, model: List, eta: float, beta: float, loss_function: str = 'mse',
                dropout: float = 1.) -> List:
    xk, d_phi = forward_pass(x0=x0, model=model)
    n = len(model)
    dC_dx = np.zeros_like(y)

    if loss_function == 'mse':
        dC_dx = 2. / y.shape[0] * (xk[n - 1] - y)
    elif loss_function == 'cross-entropy':
        dC_dx = (xk[n - 1] - y) / y.shape[0]

    for k in reversed(range(n)):

        if k == n - 1:
            dC_dx = d_phi[k] * dC_dx
        else:
            w1 = model[k + 1][0][0]
            dC_dx = d_phi[k] * np.matmul(dC_dx, w1.T)

        if k == 0:
            dx_dw = x0
        else:
            dx_dw = xk[k - 1]

        if dropout != 1.:
            mask = np.random.binomial(1., dropout, dC_dx.shape)
            dC_dx = dropout * mask * dC_dx
        dC_dw = np.matmul(dx_dw.T, dC_dx)
        dC_db = np.sum(dC_dx, axis=0).reshape(1, -1)

        (w, b) = model[k][0]
        (dw, db) = model[k][2]
        dC_db = dC_db + beta * db
        dC_dw = dC_dw + beta * dw
        b = b - eta * dC_db
        w = w - eta * dC_dw

        model[k][0] = (w, b)
        model[k][2] = (dC_dw, dC_db)

    return model


def validate_accuracy(x_test: ndarray, y_test: ndarray, model: List) -> None:
    predictions = forward(x_test, model=model)
    accuracy = np.equal(np.argmax(predictions, axis=1), y_test).sum() * 100 / y_test.shape[0]
    return print(f'''The model validation accuracy is: {accuracy:.2f}%''')
