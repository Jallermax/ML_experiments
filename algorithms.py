import json
import shutil

import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression(object):

    def __init__(self, iterations=100, learning_rate=0.001, activation_fn='sigmoid', cost_record_freq=100,
                 print_cost=False) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.cost_record_freq = cost_record_freq
        self.print_cost = print_cost
        if activation_fn == 'sigmoid':
            self.activation_fn = sigmoid
        elif activation_fn == 'relu':
            self.activation_fn = relu
        else:
            raise ValueError(f"Activation function {activation_fn} is not implemented or it has spelling error")

        self.weights = {}
        self.costs = []
        self.test_accuracy = 0

    def train(self, x_train_t, y_train_t, x_test_t, y_test_t, sample_weights: tuple = None):
        """

        :param x_train_t: train data of size (number of examples, number of features)
        :param y_train_t: train result data of size (number of examples, 1)
        :param x_test_t: train data of size (number of examples, number of features)
        :param y_test_t: test result data of size (number of examples, 1)
        :param sample_weights: initial weights to use in format ([1.2139503420, 3.536217547641585, ...], [-23.2442522])
            sample_weights.shape == (x_train.shape[0], 1)
        """
        x_train = x_train_t.T
        y_train = y_train_t.T
        x_test = x_test_t.T
        y_test = y_test_t.T
        # X -> Z = W_init * X + b_init -> a = g(Z) -> sum(loss_func(y, a >0.5))
        # Forward propagation -> -> Cost function -> Back Propagation
        self.init_weights((x_train.shape[0], 1), sample_weights)
        self._optimize_(x_train, y_train)

        # Predict test/train set examples
        y_prediction_test = self.predict(x_test)
        y_prediction_train = self.predict(x_train)

        # Print train/test Errors
        print(f"train accuracy: {100 - np.mean(np.abs(y_prediction_train - y_train)) * 100:.3f} %")
        self.test_accuracy = 100 - np.mean(np.abs(y_prediction_test - y_test)) * 100
        print(f"test accuracy: {self.test_accuracy:.3f} %")

    def predict(self, x):
        w = self.weights['w']
        b = self.weights['b']
        z = np.dot(w.T, x) + b
        a = self.activation_fn(z)
        return a > 0.5

    def save_weights(self):
        """Save weights to file"""
        model_info = {"test_accuracy": self.test_accuracy, "iter_num": self.iterations, "lr": self.learning_rate}

        original = r"models/weights.json"
        backup = r"models/weights.json.bck"
        shutil.copyfile(original, backup)
        with open('models/weights.json', 'w', encoding='utf8') as f:
            json.dump((self.weights['w'].tolist(), self.weights['b'].tolist(), model_info), f, indent=2)
        # np.savetxt("models/weights_w.csv", self.weights['w'], fmt="%s", delimiter=",")
        # np.savetxt("models/weights_b.csv", self.weights['b'], fmt="%s", delimiter=",")
        pass

    def init_weights(self, shape: tuple, weights: tuple = None):
        if weights:
            w, b = weights
            w = np.array(w)
            b = np.array(b)
        else:
            w = np.zeros(shape)
            b = np.zeros(shape[1])
        self.weights['w'] = w
        self.weights['b'] = b

    def _propagate_(self, w, b, x, y):
        m = x.shape[1]

        # Forward propagation ->
        z = np.dot(w.T, x) + b
        a = self.activation_fn(z)

        cost = np.sum(loss(a, y)) / m
        cost = np.squeeze(np.array(cost))

        dw = np.dot(x, (a - y).T) / m
        db = np.sum(a - y) / m

        return cost, dw, db

    def _optimize_(self, x, y) -> (dict, dict):
        w = self.weights['w']
        b = self.weights['b']

        for i in range(self.iterations):

            # Cost and gradient calculation
            cost, dw, db = self._propagate_(w, b, x, y)

            # Update weights
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db

            # Record the costs
            if i % self.cost_record_freq == 0:
                self.costs.append(cost)

            # Print the cost every {cost_record_freq} training examples
            if self.print_cost and i % self.cost_record_freq == 0:
                print(f"Cost after iteration {i}: {cost}")

        self.weights = {"w": w,
                        "b": b}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def loss(y_pred, y_true):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


if __name__ == "__main__":
    with open('models/weights.json', 'r', encoding='utf8') as _f:
        _weights = json.load(_f)

    # rx = np.random.random_sample((50000, 21))
    rx = 600 * np.random.random_sample((50000, 21)).astype(dtype='float64') - 289
    ry = np.array([(np.sum(r) > 11).astype(int) for r in rx])
    from sklearn.model_selection import train_test_split

    _train_x, _eval_x, _train_y, _eval_y = train_test_split(rx, ry)
    _model = LogisticRegression(iterations=100000, learning_rate=0.007, print_cost=False, cost_record_freq=20000)
    # _model = LogisticRegression(iterations=0)  # to predict using saved weight w/o training
    _model.train(_train_x, _train_y, _eval_x, _eval_y, (_weights[0], _weights[1]))
    _model.save_weights()
    _y_pred = _model.predict(np.array([
        [0, -320, 0, 0, 0, 0, 2, 0, -2, 0, 0.9, 0, 0, 0, 0, 0, 0, 12, 0, 320, -1],  # 11.9 -> True
        [0, -320, 0, 0, 0, 0, 2, 0, -2, 0, 0.2, 0, 0, 0, 0, 0, 0, 12, 0, 320, -1],  # 11.2 -> True
        [0, -320, 0, 0, 0, 0, 2, 0, -2, 0, 0.2, 0, 0, 0, 0, 0, 0, 12, 0, 319, -1],  # 10.2 -> False
        [0.5, 2.5, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1201.5, 0.5, 0.5, -1200, 0.5, 0.5, 0.5, 0.5, 0.5],   # 11 -> False
        [0.5, 2.5, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1201.501, 0.5, 0.5, -1200, 0.5, 0.5, 0.5, 0.5, 0.5],   # 11.01 -> True
        [0.5, 2.5, 0, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1201.5, 0.5, 0.5, -1200.05, 0.5, 0.5, 0.5, 0.5, 0.5],   # 10.95 -> False
    ]).T)
    print(_y_pred)
