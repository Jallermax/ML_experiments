import numpy as np


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

        self.params = {}
        self.costs = []

    def train(self, x_train_t, y_train_t, x_test_t, y_test_t):
        """

        :param x_train_t: train data of size (number of examples, number of features)
        :param y_train_t: train result data of size (number of examples, 1)
        :param x_test_t: train data of size (number of examples, number of features)
        :param y_test_t: test result data of size (number of examples, 1)
        :return:
        """
        x_train = x_train_t.T
        y_train = y_train_t.T
        x_test = x_test_t.T
        y_test = y_test_t.T
        # X -> Z = W_init * X + b_init -> a = g(Z) -> sum(loss_func(y, a >0.5))
        # Forward propagation -> -> Cost function -> Back Propagation
        w_init = np.zeros((x_train.shape[0], 1))
        b_init = 0
        self._optimize_(w_init, b_init, x_train, y_train)

        # Predict test/train set examples
        y_prediction_test = self.predict(x_test)
        y_prediction_train = self.predict(x_train)

        # Print train/test Errors
        print(f"train accuracy: {100 - np.mean(np.abs(y_prediction_train - y_train)) * 100} %")
        print(f"test accuracy: {100 - np.mean(np.abs(y_prediction_test - y_test)) * 100} %")

    def predict(self, x):
        # x = x.T
        w = self.params['w']
        b = self.params['b']
        z = np.dot(w.T, x) + b
        a = self.activation_fn(z)
        return a > 0.5

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

    def _optimize_(self, w, b, x, y) -> (dict, dict):

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

        self.params = {"w": w,
                       "b": b}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def loss(y_pred, y_true):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


if __name__ == "__main__":
    rx = np.random.random_sample((668, 21))
    # ry = np.random.random_sample((668, 1))
    ry = np.array([(np.sum(r) > 11).astype(int) for r in rx])
    from sklearn.model_selection import train_test_split
    train_x, eval_x, train_y, eval_y = train_test_split(rx, ry)

    model = LogisticRegression(iterations=10000, learning_rate=0.01)
    model.train(train_x, train_y, eval_x, eval_y)
