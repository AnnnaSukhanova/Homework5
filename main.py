import numpy as np
import plotly
import plotly.express as px
import sys
import os
from sklearn import datasets
import pickle


class Utils:
    @staticmethod
    def learn_test_validate(X, Y, lp=0.8, tp=0.1, vp=0.1):
        permutation = np.random.permutation(X.shape[0])
        Xrand = X[permutation]
        Yrand = Y[permutation]

        learnX = Xrand[0:int(lp * Xrand.shape[0])]
        testX = Xrand[int(lp * Xrand.shape[0]):int((lp + tp) * Xrand.shape[0])]
        validateX = Xrand[int((lp + tp) * Xrand.shape[0]):int((lp + tp + vp) * Xrand.shape[0])]

        learnY = Yrand[0:int(lp * Yrand.shape[0])]
        testY = Yrand[int(lp * Yrand.shape[0]):int((lp + tp) * Yrand.shape[0])]
        validateY = Yrand[int((lp + tp) * Yrand.shape[0]):int((lp + tp + vp) * Yrand.shape[0])]

        return learnX, learnY, testX, testY, validateX, validateY

    @staticmethod
    def to_one_hot(T):
        t_max = max(T)
        result = []
        for t in T:
            encoding = [1 if t == i else 0 for i in range(t_max + 1)]
            result.append(encoding)
        return np.array(result)

    @staticmethod
    def to_digits(onehotY):
        return np.argmax(onehotY, axis=1)

    @staticmethod
    def confusion_to_accuracy(confusion):
        true = 0
        for i in range(confusion.shape[0]):
            true += confusion[i, i]
        return true / np.sum(confusion)


class SoftmaxRegression:
    @classmethod
    def confusion(cls, Y, T):
        classes = Y.shape[1]
        predicted_digits = np.argmax(Y, axis=1)
        real_digits = np.argmax(T, axis=1)
        result = np.zeros((classes, classes))
        for i in range(len(predicted_digits)):
            real = real_digits[i]
            pred = predicted_digits[i]
            result[real, pred] += 1
        return result

    @classmethod
    def load(cls, filepath):
        model = None
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model

    def __init__(self, a=0.000001, max_steps=10000, epoches=1000, learning_rate=0.1):
        self._id = int(np.random.uniform(0, 1000))
        self._w = None
        self._b = None
        self._a = a
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._min_diff = 0.000001
        self._min_grad = 0.05
        self._batch_size = 300
        self._epoches = epoches

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @property
    def Id(self):
        return self._id

    @property
    def W(self):
        return self._w

    @property
    def B(self):
        return self._b

    def fit(self, X, T, Xval, Tval):
        X = self._standartize(X)
        self._w = self._init_weights((T.shape[1], X.shape[1]))
        self._b = self._init_b(T.shape[1])
        Accuracy = []
        Loss = []

        Yval, digits = self.predict(Xval)
        print(f"Confusion matrix on start = {self.confusion(Yval, Tval)}")

        steps_in_epoch = X.shape[0] // self._batch_size
        steps_in_epoch -= 1 if X.shape[0] % self._batch_size == 0 else 0
        for i in range(self._epoches):
            for step in range(steps_in_epoch):
                Xbatch, Tbatch = self._get_batch(X, T, step)
                Yprev, digits = self._predict_without_stadartize(Xbatch)

                self._w = self._w - self._learning_rate * self._gradient_w(Xbatch, Yprev, Tbatch).T
                self._b = self._b - self._learning_rate * self._gradient_b(Xbatch, Yprev, Tbatch)
                Y, digits = self._predict_without_stadartize(Xbatch)

            if i % 10 == 0:
                Accuracy.append(self._accuracy(Y, Tbatch))
                Loss.append(self._loss(Y, Tbatch))
                print(f"[{i} EPOCH]: accuracy = {self._accuracy(Y, Tbatch)}")
                print(f"[{i} EPOCH]: loss function = {self._loss(Y, Tbatch)}")
            if np.linalg.norm(np.abs(Y - Yprev)) < self._min_diff:
                print(f"Normalized difference between Y and Yprev = " +
                      f"{np.linalg.norm(np.abs(Y - Yprev))}")
                break
            if np.linalg.norm(self._gradient_w(Xbatch, Y, Tbatch)) < self._min_grad:
                print(f"Normalized gradient equal "
                      + f"{np.linalg.norm(self._gradient_w(Xbatch, Y, Tbatch))}")
                break

        Yval, digits = self.predict(Xval)
        print(f"Confusion matrix on end = {self.confusion(Yval, Tval)}")

        return self._w, Accuracy, Loss

    def _init_weights(self, shape):
        return np.zeros(shape)

    def _init_b(self, classes):
        return np.zeros(classes)

    def _standartize(self, X):
        X = np.copy(X)
        for i in range(X.shape[1]):
            if X[:, i].std() != 0:
                X[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()
        return X

    def _get_batch(self, X, T, step):
        left = (step * self._batch_size)
        right = ((step + 1) * self._batch_size)
        if right > X.shape[0]:
            right = X.shape[0]
        Xbatch = X[left: right]
        Tbatch = T[left: right]
        return Xbatch, Tbatch

    def _gradient_w(self, X, Y, T):
        result = (X.T @ (Y - T) - self._a * (self._w).T) / len(X)
        return result

    def _gradient_b(self, X, Y, T):
        return ((Y - T).T @ np.ones(Y.shape[0])) / len(Y)

    def _softmax(self, Z):
        result = []
        for i in range(Z.shape[0]):
            vector = np.array([np.e ** elem for elem in Z[i]])
            vector /= sum(vector)
            result.append(vector)
        return np.array(result)

    def accuracy(self, Y, T):
        return self._accuracy(Utils.to_one_hot(Y), Utils.to_one_hot(T))

    def _accuracy(self, Y, T):
        true = 0
        true = np.sum(np.argmax(Y, axis=1) == np.argmax(T, axis=1))
        return true / Y.shape[0]

    def _loss(self, Y, T):
        return -np.sum(np.log(Y[T == 1])) / Y.shape[0]

    def predict(self, X):
        return self._predict_without_stadartize(self._standartize(X))

    def _predict_without_stadartize(self, X):
        P = self._softmax(X @ self._w.T + self._b)
        Y = np.argmax(P, axis=1)
        return P, Y

    def test(self, X, T):
        Y, digits = self.predict(X)
        accuracy = self._accuracy(Y, T)
        return accuracy


filepath = None
if len(sys.argv) > 2:
    filepath = sys.argv[2]
    if sys.argv[1] == "load":
        print(f"Loading saved model from {filepath}")
        model = SoftmaxRegression.load(filepath)
        print(f"Successfully load model, model.id = {model.Id}")
        print(f"Exiting...")
        os._exit(0)
    elif sys.argv[1] == "save":
        print(f"Will save model after all")
    else:
        print(f"Undefined command = {sys.argv[1]}")
        print(f"Usage: {sys.argv[0]} [load|save] [filepath]")

dataset = datasets.load_digits()
X = dataset['data']
T = Utils.to_one_hot(dataset['target'])
Xlearn, Tlearn, Xtest, Ttest, Xval, Tval = Utils.learn_test_validate(X, T)
model = SoftmaxRegression()
w, accuracy, loss = model.fit(Xlearn, Tlearn, Xval, Tval)
print(f"validation accuracy = {model.test(Xval, Tval)}")

if filepath != None:
    print(f"Start saving model with model.id = {model.Id}")
    model.save(filepath)
    print(f"Model saved")

P, digits = model.predict(Xval)
real_digits = np.argmax(Tval, axis=1)
true = []
false = []
for i in range(P.shape[0]):
    if digits[i] == real_digits[i]:
        true.append(np.array((digits[i], max(P[i]), i, real_digits[i], P[i])))
    else:
        false.append(np.array((digits[i], max(P[i]), i, real_digits[i], P[i])))
true = np.array(true)
false = np.array(false)
true = true[np.argsort(-1 * true[:, 1])]
false = false[np.argsort(-1 * false[:, 1])]

for i in range(3):
    img = Xval[int(true[i][2])]
    img.resize((8, 8))
    fig1 = px.imshow(img,
                     title=f"Good recognition {i}(predicted = {true[i][0]}, real = {true[i][3]}, P = {true[i][1]})")
    plotly.offline.plot(fig1, filename=f'C:/plotly/Good recognition{2 * i}.html')
    img = Xval[int(false[i][2])]
    img.resize(8, 8)
    fig2 = px.imshow(img,
                     title=f"Bad recognition {i}(predicted = {false[i][0]}, real = {false[i][3]}, P = {false[i][1]})")
    plotly.offline.plot(fig2, filename=f'C:/plotly/Bad recognition{2 * i + 1}.html')

fig = px.line(x=np.arange(len(accuracy)) * 100, y=accuracy, title="Accuracy")
plotly.offline.plot(fig, filename=f'C:/plotly/Accuracy.html')
fig = px.line(x=np.arange(len(loss)) * 100, y=loss, title="Loss")
plotly.offline.plot(fig, filename=f'C:/plotly/Loss.html')
