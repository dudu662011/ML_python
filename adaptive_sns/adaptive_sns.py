import math
import random


def generate_dummy_data(n_samples=100, image_dim=5, tabular_dim=3):
    images = [[random.uniform(0, 1) for _ in range(image_dim)] for _ in range(n_samples)]
    tables = [[random.uniform(0, 1) for _ in range(tabular_dim)] for _ in range(n_samples)]
    labels = [random.randint(0, 1) for _ in range(n_samples)]
    return images, tables, labels


def merge_modalities(img_feats, tab_feats):
    return [i + t for i, t in zip(img_feats, tab_feats)]


class AdaptiveSquashNormSigmoid:
    """Adaptive Squash-Norm-Sigmoid normalization with trainable parameters."""

    def __init__(self, alpha=1.0, beta=0.0, gamma=1.0, epsilon=1e-8):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.mean = None
        self.std = None

    def fit(self, data):
        cols = list(zip(*data))
        self.mean = [sum(c) / len(c) for c in cols]
        self.std = []
        for j, c in enumerate(cols):
            var = sum((x - self.mean[j]) ** 2 for x in c) / len(c)
            self.std.append(max(math.sqrt(var), self.epsilon))

    def _squash(self, x):
        return math.tanh(self.alpha * x)

    def transform(self, data):
        normalized = []
        for row in data:
            tmp = []
            for j, val in enumerate(row):
                x = (val - self.mean[j]) / self.std[j]
                x = self._squash(x)
                tmp.append(x)
            normalized.append(tmp)
        # scale to 0-1 per feature
        cols = list(zip(*normalized))
        mins = [min(c) for c in cols]
        maxs = [max(c) for c in cols]
        scaled = []
        for row in normalized:
            srow = []
            for j, val in enumerate(row):
                denom = maxs[j] - mins[j] + self.epsilon
                v = (val - mins[j]) / denom
                v = 1.0 / (1.0 + math.exp(-(self.beta + self.gamma * v)))
                srow.append(v)
            scaled.append(srow)
        return scaled


class LogisticRegression:
    def __init__(self, n_features, lr=0.1):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = lr

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self._sigmoid(z)

    def fit(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict_proba(xi)
                error = yi - pred
                for j in range(len(self.weights)):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict(self, X):
        return [1 if self.predict_proba(xi) >= 0.5 else 0 for xi in X]


def accuracy(y_true, y_pred):
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def random_search(param_space, X_train, y_train, X_val, y_val, iterations=10):
    best_score = -1
    best_params = None
    for _ in range(iterations):
        alpha = random.uniform(*param_space['alpha'])
        beta = random.uniform(*param_space['beta'])
        gamma = random.uniform(*param_space['gamma'])
        norm = AdaptiveSquashNormSigmoid(alpha, beta, gamma)
        norm.fit(X_train)
        X_tr = norm.transform(X_train)
        X_v = norm.transform(X_val)
        model = LogisticRegression(len(X_tr[0]))
        model.fit(X_tr, y_train, epochs=50)
        preds = model.predict(X_v)
        score = accuracy(y_val, preds)
        if score > best_score:
            best_score = score
            best_params = (alpha, beta, gamma)
    return best_params, best_score


def train_with_best_params(X_train, y_train, X_val, y_val, best_params):
    alpha, beta, gamma = best_params
    norm = AdaptiveSquashNormSigmoid(alpha, beta, gamma)
    norm.fit(X_train + X_val)
    X_tr = norm.transform(X_train + X_val)
    model = LogisticRegression(len(X_tr[0]))
    model.fit(X_tr, y_train + y_val, epochs=100)
    return norm, model


def main():
    images, tables, labels = generate_dummy_data(150)
    data = merge_modalities(images, tables)
    # simple split
    train_end = 100
    val_end = 120
    X_train, y_train = data[:train_end], labels[:train_end]
    X_val, y_val = data[train_end:val_end], labels[train_end:val_end]
    X_test, y_test = data[val_end:], labels[val_end:]

    param_space = {'alpha': (0.5, 2.0), 'beta': (-1.0, 1.0), 'gamma': (0.5, 2.0)}
    best_params, best_score = random_search(param_space, X_train, y_train, X_val, y_val)
    print('Best params:', best_params, 'Validation accuracy:', best_score)

    norm, model = train_with_best_params(X_train, y_train, X_val, y_val, best_params)
    X_te = norm.transform(X_test)
    preds = model.predict(X_te)
    test_acc = accuracy(y_test, preds)
    print('Test accuracy:', test_acc)


if __name__ == '__main__':
    main()
